from __future__ import annotations

from typing import Any, Literal

from product.backend.schemas import TranscriptSegment


DetectionMode = Literal["realtime", "final"]

REALTIME_INSTRUCTION = (
    "현재까지의 통화 내용만 보고 보이스피싱 위험 여부를 판단하라. "
    "위험하면 1, 아니면 0만 출력하라."
)
FINAL_INSTRUCTION = (
    "다음 통화가 보이스피싱인지 정상인지 판단하라. "
    "보이스피싱이면 1, 정상이면 0만 출력하라."
)


def normalize_segment_text(text: str) -> str:
    return " ".join((text or "").split())


def render_transcript_input(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for segment in sorted(segments, key=lambda item: (item.start_sec, item.end_sec)):
        text = normalize_segment_text(segment.text)
        if text:
            lines.append(text)
    return "\n".join(lines)


def build_model_input_row(
    *,
    call_id: str,
    segments: list[TranscriptSegment],
    mode: DetectionMode,
    output: str | None = None,
) -> dict[str, Any]:
    if mode == "realtime":
        prefix_end_idx = len(segments) - 1 if segments else None
        sample_id = f"prefix_{call_id}_{len(segments):04d}"
        sample_type = "prefix"
        instruction = REALTIME_INSTRUCTION
    elif mode == "final":
        prefix_end_idx = None
        sample_id = f"conversation_{call_id}"
        sample_type = "conversation"
        instruction = FINAL_INSTRUCTION
    else:
        raise ValueError(f"Unsupported detection mode: {mode}")

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "parent_conversation_id": call_id,
        "conversation_label": None,
        "target_label": None,
        "prefix_end_idx": prefix_end_idx,
        "instruction": instruction,
        "input": render_transcript_input(segments),
    }
    if output is not None:
        row["output"] = output
    return row
