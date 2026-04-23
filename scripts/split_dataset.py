#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper. Use make_split_manifest.py or build_labeled_datasets.py.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data_pipeline_config.json"))
    parser.add_argument("--split-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_name = "make_split_manifest.py" if args.split_only else "build_labeled_datasets.py"
    script_path = Path(__file__).resolve().with_name(script_name)
    command = [sys.executable, str(script_path), "--config", str(args.config)]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
