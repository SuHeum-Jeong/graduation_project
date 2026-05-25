from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from product.backend.model_contract import build_model_input_row
from product.backend.schemas import TranscriptSegment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_jsonl", type=Path)
    parser.add_argument("--call-id", default=None)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--sleep", action="store_true")
    return parser.parse_args()


def load_segments(path: Path) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
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
    return sorted(segments, key=lambda item: (item.start_sec, item.end_sec))


def iter_realtime_rows(
    *,
    call_id: str,
    segments: list[TranscriptSegment],
    interval_sec: float,
) -> list[dict[str, Any]]:
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    if not segments:
        return []

    max_end_sec = max(segment.end_sec for segment in segments)
    tick_count = int(math.floor(max_end_sec / interval_sec))
    rows: list[dict[str, Any]] = []

    for tick_idx in range(1, tick_count + 1):
        elapsed_sec = tick_idx * interval_sec
        visible_segments = [
            segment for segment in segments if segment.end_sec <= elapsed_sec
        ]
        if not visible_segments:
            continue
        row = build_model_input_row(
            call_id=call_id,
            segments=visible_segments,
            mode="realtime",
        )
        row["sample_id"] = f"prefix_{call_id}_tick_{tick_idx:04d}"
        row["elapsed_sec"] = elapsed_sec
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    call_id = args.call_id or args.transcript_jsonl.stem
    segments = load_segments(args.transcript_jsonl)
    rows = iter_realtime_rows(
        call_id=call_id,
        segments=segments,
        interval_sec=args.interval_sec,
    )

    previous_elapsed_sec = 0.0
    for row in rows:
        elapsed_sec = float(row["elapsed_sec"])
        if args.sleep:
            time.sleep(max(elapsed_sec - previous_elapsed_sec, 0.0))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        previous_elapsed_sec = elapsed_sec

    final_row = build_model_input_row(
        call_id=call_id,
        segments=segments,
        mode="final",
    )
    final_row["elapsed_sec"] = max((segment.end_sec for segment in segments), default=0.0)
    print(json.dumps(final_row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
