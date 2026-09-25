#!/usr/bin/env python3
"""Download a reproducible GSM8K snapshot from Hugging Face as JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DATASET = "openai/gsm8k"
CONFIG = "main"
DEFAULT_REVISION = "main"
EXPECTED_SPLIT_SIZES = {"train": 7473, "test": 1319}
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def _fetch_json(url: str, retries: int = 8) -> dict[str, Any]:
    headers = {"User-Agent": "vllm-ascend-gsm8k-benchmark/1"}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt + 1 == retries:
                raise
            if error.code != 429:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after
                else min(60.0, 5.0 * (attempt + 1))
            )
            print(f"Hugging Face rate limit; retrying in {delay:g}s", flush=True)
            time.sleep(delay)
        except OSError:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30.0, 2**attempt))
    raise AssertionError("unreachable")


def _download_split(
    split: str, revision: str, batch_size: int, request_delay: float
) -> list[dict[str, str]]:
    expected = EXPECTED_SPLIT_SIZES[split]
    rows: list[dict[str, str]] = []
    for offset in range(0, expected, batch_size):
        query = urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": split,
                "offset": offset,
                "length": min(batch_size, expected - offset),
                "revision": revision,
            }
        )
        payload = _fetch_json(f"{ROWS_ENDPOINT}?{query}")
        for wrapped in payload["rows"]:
            row = wrapped["row"]
            rows.append({"question": row["question"], "answer": row["answer"]})
        print(f"{split}: {len(rows)}/{expected}", flush=True)
        if offset + batch_size < expected:
            time.sleep(request_delay)
    if len(rows) != expected:
        raise RuntimeError(f"Downloaded {len(rows)} {split} rows, expected {expected}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            encoded = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode()
            output.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=1.1)
    parser.add_argument(
        "--splits", nargs="+", choices=tuple(EXPECTED_SPLIT_SIZES), default=["train", "test"]
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100:
        parser.error("batch-size must be between 1 and 100")
    if args.request_delay < 0:
        parser.error("request-delay must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": DATASET,
        "config": CONFIG,
        "revision": args.revision,
        "source": "https://huggingface.co/datasets/openai/gsm8k",
        "splits": {},
    }
    for split in args.splits:
        rows = _download_split(
            split, args.revision, args.batch_size, args.request_delay
        )
        filename = f"{split}.jsonl"
        sha256 = _write_jsonl(args.output_dir / filename, rows)
        manifest["splits"][split] = {
            "file": filename,
            "rows": len(rows),
            "sha256": sha256,
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
