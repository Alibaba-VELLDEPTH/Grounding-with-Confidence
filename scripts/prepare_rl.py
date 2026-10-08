#!/usr/bin/env python3
"""Prepare caller-owned, video-disjoint RL train and monitor Parquet files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
CONTRACT = {"fps": 2.0, "min_pixels": 2048, "total_pixels": 8388608}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    path = Path(path).resolve(strict=True)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSONL objects: {path}")
    return rows


def identifier(value):
    if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
        raise ValueError("request_id must be a nonempty string or integer")
    return str(value)


def intervals(value, duration):
    if not isinstance(value, list) or not value:
        raise ValueError("GT must contain at least one interval")
    result = []
    for span in value:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("GT intervals must be [start, end] pairs")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in span):
            raise ValueError("GT coordinates must be numeric")
        a, b = map(float, span)
        if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration:
            raise ValueError("GT must be finite, positive-length and inside source_duration_s")
        result.append([a, b])
    return result


def video_path(value, parent):
    if not isinstance(value, str) or not value:
        raise ValueError("video_path is required")
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else parent / path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Expected a nonempty local video: {path}")
    return path


def prepare_rows(path):
    # Same shared inference prompt, without duplicating or changing its wording.
    from infer import prompt
    path = Path(path).resolve(strict=True)
    rows, seen = [], set()
    for item in read_jsonl(path):
        uid = identifier(item["request_id"])
        if uid in seen:
            raise ValueError(f"Duplicate request_id: {uid}")
        seen.add(uid)
        video = video_path(item["video_path"], path.parent)
        duration = float(item["source_duration_s"])
        if isinstance(item["source_duration_s"], bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("source_duration_s must be finite and positive")
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValueError("query must be a nonempty string")
        gt = intervals(item["gt_intervals"], duration)
        index = item.get("official_row_index", item.get("original_index"))
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError("original/official row index must be a nonnegative integer")
        rows.append({
            "data_source": "omtg_rl",
            "prompt": [{"role": "user", "content": "<video>" + prompt(item["query"].strip())}],
            "videos": [{"video": str(video), **CONTRACT}],
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, allow_nan=False)},
            "extra_info": {"query_id": uid, "official_row_index": index,
                "video_id": str(video), "duration_s": duration, "video_size_bytes": video.stat().st_size},
        })
    return rows


def split_keys(rows):
    requests, videos, content, files = set(), set(), set(), {}
    for row in rows:
        uid = identifier(row["extra_info"]["query_id"])
        if uid in requests:
            raise ValueError(f"Duplicate query ID: {uid}")
        requests.add(uid)
        path = Path(row["videos"][0]["video"]).resolve(strict=True)
        if str(path) not in videos:
            files[str(path)] = sha(path)
            content.add(files[str(path)])
        videos.add(str(path))
    return requests, videos, content, files


def exclusions(paths):
    requests, videos, content = set(), set(), set()
    for path in paths:
        path = Path(path).resolve(strict=True)
        for item in read_jsonl(path):
            if not set(item) & {"request_id", "video_path", "video_sha256"}:
                raise ValueError("Exclusion entries need request_id, video_path or video_sha256")
            if "request_id" in item:
                requests.add(identifier(item["request_id"]))
            if "video_path" in item:
                video = video_path(item["video_path"], path.parent)
                videos.add(str(video))
                content.add(sha(video))
            if "video_sha256" in item:
                value = item["video_sha256"]
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("video_sha256 must be a lowercase SHA-256 digest")
                content.add(value)
    return requests, videos, content


def validate_disjoint(train, monitor, excluded=()):
    train_keys, monitor_keys = split_keys(train), split_keys(monitor)
    if any(a & b for a, b in zip(train_keys[:3], monitor_keys[:3])):
        raise ValueError("Train/monitor request IDs or videos overlap (including identical video bytes)")
    excluded_keys = exclusions(excluded)
    for split in (train_keys, monitor_keys):
        if any(a & b for a, b in zip(split[:3], excluded_keys)):
            raise ValueError("RL split overlaps excluded calibration/test manifest")
    return {"train": train_keys[3], "monitor": monitor_keys[3]}


def parquet_schema():
    import pyarrow as pa
    return pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("videos", pa.list_(pa.struct([("video", pa.string()), ("fps", pa.float64()),
            ("min_pixels", pa.int64()), ("total_pixels", pa.int64())]))),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        ("extra_info", pa.struct([("query_id", pa.string()), ("official_row_index", pa.int64()),
            ("video_id", pa.string()), ("duration_s", pa.float64()), ("video_size_bytes", pa.int64())])),
    ])


def load_prepared(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    if not table.schema.equals(parquet_schema(), check_metadata=False) or not len(table):
        raise ValueError("Unexpected RL Parquet schema or empty split; use prepare_rl.py")
    rows = table.to_pylist()
    for row in rows:
        if row["data_source"] != "omtg_rl" or row["reward_model"]["style"] != "rule":
            raise ValueError("Only caller-owned omtg_rl rule-reward data is accepted")
        if len(row["videos"]) != 1 or {k: row["videos"][0][k] for k in CONTRACT} != CONTRACT:
            raise ValueError("RL video contract drift")
        extra = row["extra_info"]
        path = Path(row["videos"][0]["video"])
        if not path.is_absolute() or not path.is_file() or path.stat().st_size != extra["video_size_bytes"]:
            raise ValueError("Prepared video missing or changed size")
        duration = extra["duration_s"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid prepared duration")
        intervals(json.loads(row["reward_model"]["ground_truth"]), duration)
        messages = row["prompt"]
        if len(messages) != 1 or messages[0]["role"] != "user" or not messages[0]["content"].startswith("<video>"):
            raise ValueError("Invalid prepared prompt")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Caller-owned training JSONL")
    parser.add_argument("--monitor", type=Path, required=True, help="Disjoint monitoring JSONL, not calibration/test")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="JSONL request_id/video_path/video_sha256 exclusions; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error("output-dir already exists; choose a fresh directory")
    train, monitor = prepare_rows(args.train), prepare_rows(args.monitor)
    validate_disjoint(train, monitor, args.exclude_manifest)
    receipt = {"format_version": 1, "train_rows": len(train), "monitor_rows": len(monitor),
        "original_splits_available": False, "caller_owned_data": True,
        "exclusion_manifests": {str(p.resolve()): sha(p) for p in args.exclude_manifest},
        "sources": {"train": sha(args.train), "monitor": sha(args.monitor)},
        "video_contract": CONTRACT}
    if args.check_only:
        print(json.dumps({"status": "checked", **receipt}))
        return 0
    import pyarrow as pa
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", train), ("validation", monitor)):
        pq.write_table(pa.Table.from_pylist(rows, schema=parquet_schema()), output / (name + ".parquet"))
    receipt["parquet_sha256"] = {p.name: sha(p) for p in output.glob("*.parquet")}
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "prepared", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
