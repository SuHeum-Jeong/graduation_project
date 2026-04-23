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

from src.common import DEFAULT_INSTRUCTION


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_input(utterances: list[dict[str, Any]]) -> str:
    return "\n".join(f"{utt['speaker']}: {utt['text']}" for utt in utterances)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/split_manifest_v1.json"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--raw_normal_dir", type=Path, default=Path("data/raw/normal"))
    parser.add_argument("--raw_abnormal_dir", type=Path, default=Path("data/raw/abnormal"))
    parser.add_argument("--output", type=Path, default=Path("data/final/test_unlabeled.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    target_ids = [int(value) for value in manifest[args.split]]

    rows: list[dict[str, Any]] = []
    for conversation_id in sorted(target_ids):
        if str(conversation_id).startswith("1"):
            path = args.raw_normal_dir / f"raw_{conversation_id}.json"
        else:
            path = args.raw_abnormal_dir / f"raw_{conversation_id}.json"

        conversation = load_json(path)
        rows.append(
            {
                "sample_id": f"conversation_{conversation_id}",
                "sample_type": "conversation",
                "parent_conversation_id": conversation_id,
                "instruction": DEFAULT_INSTRUCTION,
                "input": render_input(conversation["utterances"]),
            }
        )

    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {"split": args.split, "rows": len(rows), "output": str(args.output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
