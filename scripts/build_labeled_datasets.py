#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CONVERSATION_INSTRUCTION = (
    "다음 통화가 보이스피싱인지 정상인지 판단하라. "
    "보이스피싱이면 1, 정상이면 0만 출력하라."
)
PREFIX_INSTRUCTION = (
    "현재까지의 통화 내용만 보고 보이스피싱 위험 여부를 판단하라. "
    "위험하면 1, 아니면 0만 출력하라."
)
LABELED_ID_PATTERN = re.compile(r"^labeled_(\d{6,7})\.json$")


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


def collect_labeled_map(*directories: Path) -> dict[int, Path]:
    labeled_map: dict[int, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("labeled_*.json")):
            match = LABELED_ID_PATTERN.match(path.name)
            if not match:
                continue
            labeled_map[int(match.group(1))] = path
    return labeled_map


def base_raw_id(conversation_id: int) -> int:
    raw_id_str = str(conversation_id)
    return int(raw_id_str[:6]) if len(raw_id_str) > 6 else conversation_id


def build_conversation_sample(conversation: dict[str, Any]) -> dict[str, Any]:
    conversation_id = int(conversation["conversation_id"])
    conversation_label = int(conversation["conversation_label"])
    return {
        "sample_id": f"conversation_{conversation_id}",
        "sample_type": "conversation",
        "parent_conversation_id": base_raw_id(conversation_id),
        "conversation_label": conversation_label,
        "target_label": conversation_label,
        "prefix_end_idx": None,
        "instruction": CONVERSATION_INSTRUCTION,
        "input": render_input(conversation["utterances"]),
        "output": str(conversation_label),
    }


def build_prefix_samples(conversation: dict[str, Any], prefix_step: int) -> list[dict[str, Any]]:
    conversation_id = int(conversation["conversation_id"])
    conversation_label = int(conversation["conversation_label"])
    utterances = conversation["utterances"]

    prefix_end_indices = list(range(prefix_step - 1, len(utterances), prefix_step))
    if prefix_end_indices and prefix_end_indices[-1] != len(utterances) - 1:
        prefix_end_indices.append(len(utterances) - 1)
    if not prefix_end_indices:
        prefix_end_indices = [len(utterances) - 1]

    rows: list[dict[str, Any]] = []
    for end_idx in prefix_end_indices:
        prefix_utterances = utterances[: end_idx + 1]
        if conversation_label == 0:
            target_label = 0
        else:
            target_label = int(any(int(utt.get("label", 0)) == 1 for utt in prefix_utterances))
        rows.append(
            {
                "sample_id": f"prefix_{conversation_id}_{end_idx:04d}",
                "sample_type": "prefix",
                "parent_conversation_id": base_raw_id(conversation_id),
                "conversation_label": conversation_label,
                "target_label": target_label,
                "prefix_end_idx": end_idx,
                "instruction": PREFIX_INSTRUCTION,
                "input": render_input(prefix_utterances),
                "output": str(target_label),
            }
        )
    return rows


def build_split_payload(
    split_name: str,
    split_ids_set: set[int],
    labeled_map: dict[int, Path],
    prefix_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    conversation_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []

    eligible_ids: list[int] = []
    for conversation_id in labeled_map:
        root_id = base_raw_id(conversation_id)
        if split_name == "train":
            if root_id in split_ids_set:
                eligible_ids.append(conversation_id)
        else:
            if conversation_id in split_ids_set and len(str(conversation_id)) == 6:
                eligible_ids.append(conversation_id)

    for conversation_id in sorted(eligible_ids):
        conversation = load_json(labeled_map[conversation_id])
        conversation_rows.append(build_conversation_sample(conversation))
        prefix_rows.extend(build_prefix_samples(conversation, prefix_step=prefix_step))

    return conversation_rows, prefix_rows, conversation_rows + prefix_rows


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

    manifest_path = Path(paths["splits_dir"]) / f"split_manifest_{config['version']}.json"
    split_manifest = load_json(manifest_path)

    labeled_map = collect_labeled_map(
        Path(paths["labeled_normal_dir"]),
        Path(paths["labeled_abnormal_dir"]),
    )

    raw_ids = {
        conversation_id
        for split_name in ("train", "valid", "test")
        for conversation_id in split_manifest[split_name]
    }
    missing_labeled_ids = sorted(conversation_id for conversation_id in raw_ids if conversation_id not in labeled_map)
    if missing_labeled_ids:
        raise ValueError(
            "Missing labeled files for raw conversations: "
            + ", ".join(str(conversation_id) for conversation_id in missing_labeled_ids[:20])
        )

    for split_name in ("train", "valid", "test"):
        conversation_rows, prefix_rows, final_rows = build_split_payload(
            split_name=split_name,
            split_ids_set=set(split_manifest[split_name]),
            labeled_map=labeled_map,
            prefix_step=int(config["prefix_step"]),
        )
        write_jsonl(
            Path(paths["samples_conversation_dir"]) / f"conversation_{split_name}_{config['version']}.jsonl",
            conversation_rows,
        )
        write_jsonl(
            Path(paths["samples_prefix_dir"]) / f"prefix_{split_name}_{config['version']}.jsonl",
            prefix_rows,
        )
        write_jsonl(Path(paths["final_dir"]) / f"{split_name}.jsonl", final_rows)
        print(
            json.dumps(
                {
                    "split": split_name,
                    "conversation_samples": len(conversation_rows),
                    "prefix_samples": len(prefix_rows),
                    "final_samples": len(final_rows),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
