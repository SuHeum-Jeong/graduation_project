from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from product.backend.schemas import TranscriptSegment


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(row)
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def load_transcript_segments(path: Path) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for line_number, row in enumerate(load_jsonl(path), start=1):
        try:
            segments.append(
                TranscriptSegment(
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    text=str(row["text"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{line_number} missing field {exc}") from exc
    return sorted(segments, key=lambda segment: (segment.start_sec, segment.end_sec))


def write_transcript_segments(path: Path, segments: list[TranscriptSegment]) -> None:
    write_jsonl(path, [segment.to_dict() for segment in segments])
