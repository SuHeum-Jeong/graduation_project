#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import calc_binary_metrics, get_row_label, make_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_eval_config.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
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


def trim_prompt_text(
    tokenizer: Any,
    prompt: str,
    max_prompt_tokens: int,
) -> tuple[str, bool]:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) <= max_prompt_tokens:
        return prompt, False

    marker = "[통화 내용]\n"
    answer_header = "\n\n[답변]\n0 또는 1"
    if marker not in prompt or answer_header not in prompt:
        trimmed_tokens = prompt_tokens[:max_prompt_tokens]
        return tokenizer.decode(trimmed_tokens, skip_special_tokens=True), True

    prefix, remaining = prompt.split(marker, 1)
    conversation_text, suffix = remaining.split(answer_header, 1)
    prefix = f"{prefix}{marker}"
    suffix = f"{answer_header}{suffix}"

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    ellipsis = "\n[중략]\n"
    ellipsis_tokens = tokenizer.encode(ellipsis, add_special_tokens=False)

    available_tokens = max_prompt_tokens - len(prefix_tokens) - len(suffix_tokens)
    if available_tokens <= 0:
        trimmed_tokens = (prefix_tokens + suffix_tokens)[:max_prompt_tokens]
        return tokenizer.decode(trimmed_tokens, skip_special_tokens=True), True

    conversation_tokens = tokenizer.encode(conversation_text, add_special_tokens=False)
    if len(conversation_tokens) <= available_tokens:
        return prompt, False

    if available_tokens <= len(ellipsis_tokens):
        kept_tokens = conversation_tokens[:available_tokens]
        trimmed_prompt = f"{prefix}{tokenizer.decode(kept_tokens, skip_special_tokens=True)}{suffix}"
        return trimmed_prompt, True

    kept_budget = available_tokens - len(ellipsis_tokens)
    head_budget = kept_budget // 2
    tail_budget = kept_budget - head_budget
    head_tokens = conversation_tokens[:head_budget]
    tail_tokens = conversation_tokens[-tail_budget:] if tail_budget > 0 else []
    trimmed_prompt = (
        f"{prefix}"
        f"{tokenizer.decode(head_tokens, skip_special_tokens=True)}"
        f"{ellipsis}"
        f"{tokenizer.decode(tail_tokens, skip_special_tokens=True)}"
        f"{suffix}"
    )
    return trimmed_prompt, True


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


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl(dataset_path)
    max_samples = config.get("max_samples")
    if isinstance(max_samples, int):
        dataset = dataset[:max_samples]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            "transformers/torch/peft가 설치되어 있지 않습니다. "
            "환경 준비 후 다시 실행하세요."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype_name = config.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, torch_dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch_dtype,
        device_map=config.get("device_map", "auto"),
    )
    adapter_path = config.get("adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model_input_device = next(model.parameters()).device

    y_true: list[str] = []
    y_pred: list[str] = []
    prediction_rows: list[dict[str, Any]] = []
    batch_size = max(1, int(config.get("batch_size", 1)))
    candidate_labels = ["0", "1"]
    max_length = resolve_effective_max_length(tokenizer, config)
    label_token_reserve = max(8, int(config.get("label_token_reserve", 16)))
    max_prompt_tokens = max(1, max_length - label_token_reserve)

    for start_idx in range(0, len(dataset), batch_size):
        batch_rows = dataset[start_idx : start_idx + batch_size]
        raw_prompt_texts = [
            build_chat_prompt(
                tokenizer,
                make_prompt(row.get("instruction", ""), row.get("input", "")),
            )
            for row in batch_rows
        ]
        trimmed_prompt_pairs = [
            trim_prompt_text(tokenizer, prompt_text, max_prompt_tokens)
            for prompt_text in raw_prompt_texts
        ]
        prompt_texts = [prompt_text for prompt_text, _ in trimmed_prompt_pairs]
        label_scores = score_candidate_labels(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            candidate_labels=candidate_labels,
            model_input_device=model_input_device,
            max_length=max_length,
        )
        for row, score_map, (_, was_truncated), prompt_text in zip(
            batch_rows,
            label_scores,
            trimmed_prompt_pairs,
            prompt_texts,
        ):
            gold = get_row_label(row)
            prediction = max(candidate_labels, key=lambda label: score_map.get(label, float("-inf")))
            if gold in {"0", "1"}:
                y_true.append(gold)
                y_pred.append(prediction)
            prediction_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "sample_type": row["sample_type"],
                    "parent_conversation_id": row.get("parent_conversation_id"),
                    "gold": gold or None,
                    "prediction": prediction,
                    "raw_prediction_text": prediction,
                    "label_scores": score_map,
                    "prompt_truncated": was_truncated,
                    "prompt_char_length": len(prompt_text),
                    "correct": prediction == gold if gold in {"0", "1"} else None,
                }
            )

    metrics = calc_binary_metrics(y_true, y_pred)
    truncated_count = sum(1 for row in prediction_rows if row["prompt_truncated"])
    report = {
        "model_name": config["model_name_or_path"],
        "adapter_path": adapter_path,
        "test_file": str(dataset_path),
        "prediction_mode": "label_scoring",
        "max_length": max_length,
        "label_token_reserve": label_token_reserve,
        "truncated_prompt_count": truncated_count,
        "num_rows": len(prediction_rows),
        "labeled_rows": len(y_true),
        "unlabeled_rows": len(prediction_rows) - len(y_true),
        "metrics": metrics,
        "examples": prediction_rows[:20],
    }

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for row in prediction_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with (output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
