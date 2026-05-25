#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_LABELS = {"0", "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/finetuned_eval_qwen35_2b_base/predictions.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/finetuned_eval_qwen35_2b_base"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_idx, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a partially written tail line while evaluation is still running.
                break
    return rows


def build_confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid = unlabeled = 0
    labeled = 0

    for row in rows:
        gold = row.get("gold")
        prediction = row.get("prediction")
        if gold not in VALID_LABELS:
            unlabeled += 1
            continue

        labeled += 1
        if prediction not in VALID_LABELS:
            invalid += 1
        elif gold == "1" and prediction == "1":
            tp += 1
        elif gold == "0" and prediction == "0":
            tn += 1
        elif gold == "0" and prediction == "1":
            fp += 1
        elif gold == "1" and prediction == "0":
            fn += 1

    accuracy = (tp + tn) / labeled if labeled else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "labels": ["0", "1"],
        "axis": {
            "rows": "actual",
            "columns": "predicted",
        },
        "matrix": {
            "actual_0": {
                "pred_0": tn,
                "pred_1": fp,
            },
            "actual_1": {
                "pred_0": fn,
                "pred_1": tp,
            },
        },
        "counts": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "invalid": invalid,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "total_rows": len(rows),
        },
        "metrics": {
            "accuracy": accuracy,
            "precision_label_1": precision,
            "recall_label_1": recall,
            "f1_label_1": f1,
        },
    }


def write_markdown(matrix: dict[str, Any], output_path: Path) -> None:
    counts = matrix["counts"]
    metrics = matrix["metrics"]
    cells = matrix["matrix"]
    text = (
        "# Confusion Matrix\n\n"
        "Rows are actual labels; columns are predicted labels.\n\n"
        "| actual \\\\ predicted | 0 | 1 |\n"
        "| --- | ---: | ---: |\n"
        f"| 0 | {cells['actual_0']['pred_0']} | {cells['actual_0']['pred_1']} |\n"
        f"| 1 | {cells['actual_1']['pred_0']} | {cells['actual_1']['pred_1']} |\n\n"
        f"- labeled: {counts['labeled']}\n"
        f"- invalid: {counts['invalid']}\n"
        f"- accuracy: {metrics['accuracy']:.6f}\n"
        f"- precision_label_1: {metrics['precision_label_1']:.6f}\n"
        f"- recall_label_1: {metrics['recall_label_1']:.6f}\n"
        f"- f1_label_1: {metrics['f1_label_1']:.6f}\n"
    )
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.predictions)
    matrix = build_confusion_matrix(rows)

    json_path = args.output_dir / "confusion_matrix.json"
    markdown_path = args.output_dir / "confusion_matrix.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(matrix, file, ensure_ascii=False, indent=2)
    write_markdown(matrix, markdown_path)

    print(json.dumps(matrix, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
