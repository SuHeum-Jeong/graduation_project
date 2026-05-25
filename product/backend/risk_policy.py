from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from product.backend.schemas import AlertEvent, ModelPrediction


@dataclass
class RiskPolicy:
    realtime_warning_threshold: float = 0.80
    realtime_caution_threshold: float = 0.65
    realtime_caution_consecutive: int = 2
    final_warning_threshold: float = 0.50

    def __post_init__(self) -> None:
        self._consecutive_caution_count = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RiskPolicy":
        return cls(
            realtime_warning_threshold=float(config.get("realtime_warning_threshold", 0.80)),
            realtime_caution_threshold=float(config.get("realtime_caution_threshold", 0.65)),
            realtime_caution_consecutive=int(config.get("realtime_caution_consecutive", 2)),
            final_warning_threshold=float(config.get("final_warning_threshold", 0.50)),
        )

    def evaluate_realtime(self, prediction: ModelPrediction) -> AlertEvent | None:
        if prediction.risk_score >= self.realtime_warning_threshold:
            self._consecutive_caution_count += 1
            return AlertEvent(
                event_type="realtime_warning",
                sample_id=prediction.sample_id,
                elapsed_sec=prediction.elapsed_sec,
                alert_level="warning",
                message="실시간 판정에서 보이스피싱 고위험 통화로 감지되었습니다.",
                prediction=prediction.prediction,
                risk_score=prediction.risk_score,
            )

        if prediction.risk_score >= self.realtime_caution_threshold:
            self._consecutive_caution_count += 1
            if self._consecutive_caution_count >= self.realtime_caution_consecutive:
                return AlertEvent(
                    event_type="realtime_caution",
                    sample_id=prediction.sample_id,
                    elapsed_sec=prediction.elapsed_sec,
                    alert_level="caution",
                    message="실시간 판정에서 보이스피싱 의심 신호가 연속 감지되었습니다.",
                    prediction=prediction.prediction,
                    risk_score=prediction.risk_score,
                )
            return None

        self._consecutive_caution_count = 0
        return None

    def evaluate_final(self, prediction: ModelPrediction) -> AlertEvent | None:
        if prediction.risk_score < self.final_warning_threshold:
            return None
        return AlertEvent(
            event_type="final_warning",
            sample_id=prediction.sample_id,
            elapsed_sec=prediction.elapsed_sec,
            alert_level="warning",
            message="통화 종료 후 최종 판정에서 보이스피싱 의심 전화로 판단되었습니다.",
            prediction=prediction.prediction,
            risk_score=prediction.risk_score,
        )
