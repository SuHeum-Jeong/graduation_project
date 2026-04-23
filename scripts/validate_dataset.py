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

from src.common import get_row_label


REQUIRED_FIELDS = {
    "sample_id",
    "sample_type",
    "parent_conversation_id",
    "conversation_label",
    "target_label",
    "prefix_end_idx",
    "instruction",
    "input",
    "output",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_parent_ids(rows: list[dict[str, Any]]) -> set[int]:
    return {
        int(row["parent_conversation_id"])
        for row in rows
        if isinstance(row.get("parent_conversation_id"), int)
    }


def validate_split(name: str, path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    issues: list[str] = []
    sample_types: set[str] = set()

    for index, row in enumerate(rows):
        missing_fields = sorted(REQUIRED_FIELDS - set(row.keys()))
        if missing_fields:
            issues.append(f"row={index}: missing_fields={missing_fields}")
            continue

        sample_type = str(row.get("sample_type", ""))
        sample_types.add(sample_type)
        if sample_type not in {"conversation", "prefix"}:
            issues.append(f"row={index}: invalid sample_type={sample_type}")

        label = get_row_label(row)
        if label not in {"0", "1"}:
            issues.append(f"row={index}: invalid label from output/target_label/conversation_label")

        output_label = get_row_label({"output": row.get("output")})
        target_label = get_row_label({"target_label": row.get("target_label")})
        if output_label and target_label and output_label != target_label:
            issues.append(f"row={index}: output({output_label}) != target_label({target_label})")

        if sample_type == "conversation" and row.get("prefix_end_idx") is not None:
            issues.append(f"row={index}: conversation requires prefix_end_idx=null")
        if sample_type == "prefix" and row.get("prefix_end_idx") is None:
            issues.append(f"row={index}: prefix requires prefix_end_idx not null")

    return {
        "name": name,
        "path": str(path),
        "rows": len(rows),
        "sample_types": sorted(sample_types),
        "issues": issues,
        "parent_ids": collect_parent_ids(rows),
    }


def validate_manifest(manifest_path: Path, split_results: dict[str, dict[str, Any]]) -> list[str]:
    manifest = load_json(manifest_path)
    issues: list[str] = []
    for split_name in ("train", "valid", "test"):
        if split_name not in manifest:
            issues.append(f"manifest missing key: {split_name}")
            continue

        expected_ids = set(int(value) for value in manifest[split_name])
        actual_ids = split_results[split_name]["parent_ids"]
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)

        if missing:
            issues.append(f"{split_name}: missing parent_conversation_id from file: {missing[:20]}")
        if extra:
            issues.append(f"{split_name}: extra parent_conversation_id not in manifest: {extra[:20]}")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--valid_file", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {
        "train": validate_split("train", args.train_file),
        "valid": validate_split("valid", args.valid_file),
        "test": validate_split("test", args.test_file),
    }

    all_issues: list[str] = []
    for split_name in ("train", "valid", "test"):
        all_issues.extend(f"{split_name}: {issue}" for issue in results[split_name]["issues"])

    manifest_issues: list[str] = []
    if args.split_manifest:
        manifest_issues = validate_manifest(args.split_manifest, results)
        all_issues.extend(manifest_issues)

    summary = {
        split_name: {
            "path": results[split_name]["path"],
            "rows": results[split_name]["rows"],
            "sample_types": results[split_name]["sample_types"],
            "issues": len(results[split_name]["issues"]),
        }
        for split_name in ("train", "valid", "test")
    }
    summary["manifest_checked"] = bool(args.split_manifest)
    summary["manifest_issues"] = len(manifest_issues)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if all_issues:
        print("\n[DATA VALIDATION ERRORS]")
        for issue in all_issues[:200]:
            print(f"- {issue}")
        raise SystemExit(1)

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
