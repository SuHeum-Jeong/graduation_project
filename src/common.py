import re
from typing import Any


DEFAULT_INSTRUCTION = (
    "다음 통화 내용을 읽고 보이스피싱 여부를 분류하라."
)


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in {"1", "FRAUD", "PHISHING", "SCAM", "RISK", "HIGH"}:
        return "1"
    if text in {"0", "SAFE", "NORMAL", "LOW"}:
        return "0"
    if re.search(r"FRAUD|PHISH|SCAM|RISK", text):
        return "1"
    if re.search(r"SAFE|NORMAL|LOW", text):
        return "0"
    return ""


def get_row_label(row: dict[str, Any]) -> str:
    for key in ("output", "target_label", "conversation_label"):
        if key in row:
            label = normalize_label(row.get(key))
            if label in {"0", "1"}:
                return label
    return ""


def parse_prediction(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    direct_match = re.search(r"\b([01])\b", normalized)
    if direct_match:
        return direct_match.group(1)

    compact = normalized.replace(" ", "")
    if compact.startswith("0") or compact.startswith("1"):
        return compact[0]

    upper_text = normalized.upper()
    if re.search(r"FRAUD|PHISH|SCAM|RISK", upper_text):
        return "1"
    if re.search(r"SAFE|NORMAL|LOW", upper_text):
        return "0"
    return ""


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


def calc_binary_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float | int]:
    pairs = [(truth, pred) for truth, pred in zip(y_true, y_pred) if truth in {"0", "1"}]
    if not pairs:
        return {
            "accuracy": 0.0,
            "precision_label_1": 0.0,
            "recall_label_1": 0.0,
            "f1_label_1": 0.0,
            "support": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "invalid_count": 0,
            "invalid_rate": 0.0,
        }

    total = len(pairs)
    tp = sum(1 for truth, pred in pairs if truth == "1" and pred == "1")
    tn = sum(1 for truth, pred in pairs if truth == "0" and pred == "0")
    fp = sum(1 for truth, pred in pairs if truth == "0" and pred == "1")
    fn = sum(1 for truth, pred in pairs if truth == "1" and pred == "0")
    invalid_positive = sum(1 for truth, pred in pairs if truth == "1" and pred not in {"0", "1"})
    invalid_negative = sum(1 for truth, pred in pairs if truth == "0" and pred not in {"0", "1"})
    invalid_count = invalid_positive + invalid_negative

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn + invalid_positive) if (tp + fn + invalid_positive) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision_label_1": precision,
        "recall_label_1": recall,
        "f1_label_1": f1,
        "support": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / total if total else 0.0,
    }


def make_prompt(instruction: str, user_input: str) -> str:
    safe_instruction = (instruction or "").strip() or DEFAULT_INSTRUCTION
    safe_input = (user_input or "").strip()
    return (
        f"{safe_instruction}\n\n"
        "[분류 기준]\n"
        "- 정상적인 고객센터 상담, 카드 승인/취소 확인, 일반 금융 문의는 0\n"
        "- 기관 사칭, 대출 빙자, 계좌/통장 요구, 송금 유도, 앱 설치 유도 등 금융사기 정황은 1\n"
        "- 애매하면 통화의 전체 목적과 상대방의 요구 행동을 기준으로 더 가까운 쪽 하나를 고른다\n\n"
        f"[통화 내용]\n{safe_input}\n\n"
        "[답변]\n"
        "0 또는 1"
    )
