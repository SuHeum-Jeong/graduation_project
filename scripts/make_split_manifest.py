#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RAW_ID_PATTERN = re.compile(r"^raw_(\d{6})\.json$")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def discover_raw_conversations(directory: Path, label: int) -> list[dict[str, int]]:
    conversations: list[dict[str, int]] = []
    for path in sorted(directory.glob("raw_*.json")):
        match = RAW_ID_PATTERN.match(path.name)
        if not match:
            continue
        conversations.append(
            {
                "conversation_id": int(match.group(1)),
                "conversation_label": label,
            }
        )
    return conversations


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train_ratio, valid_ratio, test_ratio = ratios
    if round(train_ratio + valid_ratio + test_ratio, 10) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    test_count = total - train_count - valid_count
    return train_count, valid_count, test_count


def split_ids(
    rows: list[dict[str, int]],
    ratios: tuple[float, float, float],
    seed: int,
    stratify_by_label: bool,
) -> dict[str, list[int]]:
    if not rows:
        raise ValueError("No raw conversations were found. Check data/raw directories.")

    rng = random.Random(seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    if stratify_by_label:
        for row in rows:
            buckets[row["conversation_label"]].append(row["conversation_id"])
    else:
        buckets[-1] = [row["conversation_id"] for row in rows]

    split_map = {"train": [], "valid": [], "test": []}
    for ids in buckets.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        train_count, valid_count, _ = allocate_counts(len(ids), ratios)
        split_map["train"].extend(ids[:train_count])
        split_map["valid"].extend(ids[train_count : train_count + valid_count])
        split_map["test"].extend(ids[train_count + valid_count :])

    for key in split_map:
        split_map[key] = sorted(split_map[key])
    return split_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_pipeline_config.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    paths = config["paths"]

    raw_rows = discover_raw_conversations(Path(paths["raw_normal_dir"]), label=0)
    raw_rows += discover_raw_conversations(Path(paths["raw_abnormal_dir"]), label=1)

    split_manifest = split_ids(
        rows=raw_rows,
        ratios=(config["train_ratio"], config["valid_ratio"], config["test_ratio"]),
        seed=int(config["seed"]),
        stratify_by_label=bool(config.get("stratify_by_label", True)),
    )

    output_path = Path(paths["splits_dir"]) / f"split_manifest_{config['version']}.json"
    write_json(output_path, split_manifest)

    counts = {
        split_name: len(split_manifest[split_name]) for split_name in ("train", "valid", "test")
    }
    print(json.dumps({"output": str(output_path), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
