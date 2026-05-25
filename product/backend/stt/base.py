from __future__ import annotations

from pathlib import Path
from typing import Protocol

from product.backend.schemas import TranscriptSegment


class AudioTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        ...
