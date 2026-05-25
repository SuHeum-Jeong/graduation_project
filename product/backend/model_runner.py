from __future__ import annotations

import math
from typing import Any, Protocol

from product.backend.schemas import ModelPrediction
from src.common import make_prompt, trim_prompt_text


CANDIDATE_LABELS = ["0", "1"]


class ModelRunner(Protocol):
    def predict(self, row: dict[str, Any]) -> ModelPrediction:
        ...


def resolve_chat_template_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "enable_thinking" in config:
        kwargs["enable_thinking"] = bool(config["enable_thinking"])
    return kwargs


def build_chat_prompt(tokenizer: Any, prompt: str, config: dict[str, Any]) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [
        {
            "role": "system",
            "content": (
                "너는 보이스피싱 통화 여부를 판별하는 이진 분류기다. "
                "입력된 통화가 보이스피싱이 아니면 0, 보이스피싱이면 1을 출력하라. "
                "반드시 0 또는 1 한 글자만 출력하고 다른 설명은 금지한다."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    chat_template_kwargs = resolve_chat_template_kwargs(config)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def resolve_effective_max_length(tokenizer: Any, config: dict[str, Any]) -> int:
    config_max_length = config.get("max_length")
    if isinstance(config_max_length, int) and config_max_length > 0:
        return config_max_length

    model_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 100000:
        return 4096
    return model_max_length


def score_candidate_labels(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    candidate_labels: list[str],
    model_input_device: Any,
    max_length: int | None = None,
) -> list[dict[str, float]]:
    import torch

    prompt_encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1).tolist()
    scores_by_row = [dict() for _ in prompt_texts]

    label_token_ids: dict[str, int] = {}
    for label in candidate_labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            label_token_ids = {}
            break
        label_token_ids[label] = token_ids[0]

    if label_token_ids:
        encoded = {key: value.to(model_input_device) for key, value in prompt_encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits

        next_token_log_probs = torch.log_softmax(logits, dim=-1)
        for row_idx, prompt_length in enumerate(prompt_lengths):
            next_token_pos = max(int(prompt_length) - 1, 0)
            row_log_probs = next_token_log_probs[row_idx, next_token_pos]
            for label, token_id in label_token_ids.items():
                scores_by_row[row_idx][label] = float(row_log_probs[token_id].item())
        return scores_by_row

    for label in candidate_labels:
        full_texts = [f"{prompt}{label}" for prompt in prompt_texts]
        encoded = tokenizer(
            full_texts,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model_input_device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        shifted_input_ids = encoded["input_ids"][:, 1:]
        shifted_attention_mask = encoded["attention_mask"][:, 1:]

        for row_idx, prompt_length in enumerate(prompt_lengths):
            label_start = max(int(prompt_length) - 1, 0)
            valid_positions = torch.nonzero(shifted_attention_mask[row_idx], as_tuple=False).squeeze(-1)
            label_positions = valid_positions[valid_positions >= label_start]
            if label_positions.numel() == 0:
                scores_by_row[row_idx][label] = float("-inf")
                continue

            token_ids = shifted_input_ids[row_idx, label_positions]
            token_log_probs = log_probs[row_idx, label_positions, token_ids]
            scores_by_row[row_idx][label] = float(token_log_probs.sum().item())

    return scores_by_row


def scores_to_risk_score(score_map: dict[str, float]) -> float:
    score_0 = score_map.get("0", float("-inf"))
    score_1 = score_map.get("1", float("-inf"))
    if not math.isfinite(score_0) and not math.isfinite(score_1):
        return 0.0
    max_score = max(score_0, score_1)
    exp_0 = math.exp(score_0 - max_score) if math.isfinite(score_0) else 0.0
    exp_1 = math.exp(score_1 - max_score) if math.isfinite(score_1) else 0.0
    denominator = exp_0 + exp_1
    return exp_1 / denominator if denominator else 0.0


class QwenLabelScoringRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen 모델 실행에 필요한 torch/transformers가 없습니다. "
                "product/requirements.txt를 설치한 뒤 다시 실행하세요."
            ) from exc

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        torch_dtype_name = config.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, torch_dtype_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            torch_dtype=torch_dtype,
            device_map=config.get("device_map", "auto"),
        )

        adapter_path = config.get("adapter_path")
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA adapter를 쓰려면 peft가 필요합니다. "
                    "product/requirements.txt를 설치한 뒤 다시 실행하세요."
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self.model_input_device = next(self.model.parameters()).device
        self.max_length = resolve_effective_max_length(self.tokenizer, config)
        label_token_reserve = max(8, int(config.get("label_token_reserve", 16)))
        self.max_prompt_tokens = max(1, self.max_length - label_token_reserve)

    def predict(self, row: dict[str, Any]) -> ModelPrediction:
        raw_prompt = make_prompt(row.get("instruction", ""), row.get("input", ""))
        chat_prompt = build_chat_prompt(self.tokenizer, raw_prompt, self.config)
        prompt_text, was_truncated = trim_prompt_text(
            self.tokenizer,
            chat_prompt,
            self.max_prompt_tokens,
        )
        score_map = score_candidate_labels(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_texts=[prompt_text],
            candidate_labels=CANDIDATE_LABELS,
            model_input_device=self.model_input_device,
            max_length=self.max_length,
        )[0]
        prediction = max(CANDIDATE_LABELS, key=lambda label: score_map.get(label, float("-inf")))
        risk_score = scores_to_risk_score(score_map)
        return ModelPrediction(
            sample_id=str(row["sample_id"]),
            sample_type=str(row["sample_type"]),
            parent_conversation_id=str(row["parent_conversation_id"]),
            elapsed_sec=float(row["elapsed_sec"]) if row.get("elapsed_sec") is not None else None,
            prediction=prediction,
            risk_score=risk_score,
            label_scores=score_map,
            prompt_truncated=was_truncated,
            prompt_char_length=len(prompt_text),
        )


class KeywordMockRunner:
    """Small local runner for smoke tests when model dependencies are absent."""

    HIGH_RISK_KEYWORDS = (
        "검찰",
        "수사",
        "범죄",
        "계좌",
        "송금",
        "안전 계좌",
        "앱",
        "비밀번호",
        "인증번호",
    )

    def predict(self, row: dict[str, Any]) -> ModelPrediction:
        input_text = str(row.get("input", ""))
        hits = sum(1 for keyword in self.HIGH_RISK_KEYWORDS if keyword in input_text)
        risk_score = min(0.05 + hits * 0.15, 0.98)
        prediction = "1" if risk_score >= 0.5 else "0"
        return ModelPrediction(
            sample_id=str(row["sample_id"]),
            sample_type=str(row["sample_type"]),
            parent_conversation_id=str(row["parent_conversation_id"]),
            elapsed_sec=float(row["elapsed_sec"]) if row.get("elapsed_sec") is not None else None,
            prediction=prediction,
            risk_score=risk_score,
            label_scores={
                "0": math.log(max(1.0 - risk_score, 1e-8)),
                "1": math.log(max(risk_score, 1e-8)),
            },
            prompt_truncated=False,
            prompt_char_length=len(input_text),
        )


def build_model_runner(config: dict[str, Any], *, mock_model: bool = False) -> ModelRunner:
    if mock_model:
        return KeywordMockRunner()
    return QwenLabelScoringRunner(config)
