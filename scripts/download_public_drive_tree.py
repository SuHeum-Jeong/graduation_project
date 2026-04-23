#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


FOLDER_RE = re.compile(
    r'<div class="flip-entry" id="entry-([^"]+)".*?<a href="https://drive.google.com/drive/folders/[^"]+".*?<div class="flip-entry-title">([^<]+)</div>',
    re.DOTALL,
)
FILE_RE = re.compile(
    r'<div class="flip-entry" id="entry-([^"]+)".*?<a href="https://drive.google.com/file/d/[^"]+".*?<div class="flip-entry-title">([^<]+)</div>',
    re.DOTALL,
)


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_embedded_entries(folder_id: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    text = fetch_text(url)
    folders = [(entry_id, html.unescape(title)) for entry_id, title in FOLDER_RE.findall(text)]
    files = [(entry_id, html.unescape(title)) for entry_id, title in FILE_RE.findall(text)]
    return folders, files


def download_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, output_path.open("wb") as file:
        file.write(response.read())


def walk_and_download(folder_id: str, output_dir: Path, stats: dict[str, int]) -> None:
    folders, files = parse_embedded_entries(folder_id)
    for file_id, name in files:
        target_path = output_dir / name
        if target_path.exists():
            stats["skipped"] += 1
            continue
        download_file(file_id, target_path)
        stats["downloaded"] += 1
        print(f"downloaded {target_path}", flush=True)

    for child_folder_id, child_name in folders:
        child_output = output_dir / child_name
        child_output.mkdir(parents=True, exist_ok=True)
        stats["folders"] += 1
        print(f"entering {child_output}", flush=True)
        walk_and_download(child_folder_id, child_output, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_id", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = {"downloaded": 0, "skipped": 0, "folders": 0}
    try:
        walk_and_download(args.folder_id, args.output_dir, stats)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Download failed: {exc}") from exc
    print(stats)


if __name__ == "__main__":
    main()
