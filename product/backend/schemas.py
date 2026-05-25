from __future__ import annotations

from typing import Any

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
        }


@dataclass(frozen=True)
class ModelPrediction:
    sample_id: str
    sample_type: str
    parent_conversation_id: str
    elapsed_sec: float | None
    prediction: str
    risk_score: float
    label_scores: dict[str, float]
    prompt_truncated: bool
    prompt_char_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample_type": self.sample_type,
            "parent_conversation_id": self.parent_conversation_id,
            "elapsed_sec": self.elapsed_sec,
            "prediction": self.prediction,
            "risk_score": self.risk_score,
            "label_scores": self.label_scores,
            "prompt_truncated": self.prompt_truncated,
            "prompt_char_length": self.prompt_char_length,
        }


@dataclass(frozen=True)
class AlertEvent:
    event_type: str
    sample_id: str
    elapsed_sec: float | None
    alert_level: str
    message: str
    prediction: str
    risk_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "sample_id": self.sample_id,
            "elapsed_sec": self.elapsed_sec,
            "alert_level": self.alert_level,
            "message": self.message,
            "prediction": self.prediction,
            "risk_score": self.risk_score,
        }
