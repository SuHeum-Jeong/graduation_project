from __future__ import annotations

from pathlib import Path
from typing import Any

from product.backend.schemas import TranscriptSegment


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_size_or_path: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = "ko",
        vad_filter: bool = True,
        beam_size: int = 5,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "녹음 파일 STT에는 faster-whisper가 필요합니다. "
                "product/requirements.txt를 설치하거나 --transcript-jsonl로 전사 결과를 넘겨주세요."
            ) from exc

        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FasterWhisperTranscriber":
        return cls(
            model_size_or_path=str(config.get("model_size_or_path", "small")),
            device=str(config.get("device", "auto")),
            compute_type=str(config.get("compute_type", "default")),
            language=config.get("language", "ko"),
            vad_filter=bool(config.get("vad_filter", True)),
            beam_size=int(config.get("beam_size", 5)),
        )

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        segments, _ = self.model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
            word_timestamps=False,
        )
        transcript_segments: list[TranscriptSegment] = []
        for segment in segments:
            text = " ".join(str(segment.text).split())
            if not text:
                continue
            transcript_segments.append(
                TranscriptSegment(
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    text=text,
                )
            )
        return transcript_segments
