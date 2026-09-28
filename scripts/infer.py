#!/usr/bin/env python3
"""Run the released generator and confidence head on local videos."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import _infer_core as core
from tvg_confidence.common import sha256_file
from tvg_confidence.selection import pool
from video_budget import VIDEO_CONTRACT, install_into_core


def prompt(query):
    return (
        "You are given a video.\nTask: temporal retrieval candidate generation.\n"
        f'Find time spans in which the query "{query}" occurs.\n'
        "Build a high-recall candidate pool for a later confidence selector. "
        "First cover distinct occurrences throughout the video. Then include a few "
        "plausible alternative boundaries when useful. Overlapping alternatives are allowed, "
        "but avoid near-identical intervals and never repeat identical coordinates. "
        "Return at most 32 candidates in seconds. Do not fill the "
        "budget with unrelated intervals. Close the array after the useful candidates.\n"
        "Output only one JSON array: [[start, end], ...]."
    )


def prepare_rows(input_path):
    rows, seen = [], set()
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        uid = str(item["request_id"])
        if uid in seen:
            raise ValueError(f"Duplicate request_id: {uid}")
        seen.add(uid)
        video = Path(item["video_path"]).expanduser()
        if not video.is_absolute():
            video = input_path.parent / video
        video = video.resolve(strict=True)
        duration = float(item["source_duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid source_duration_s: {uid}")
        query = str(item["query"]).strip()
        if not query:
            raise ValueError(f"Empty query: {uid}")
        rows.append({
            **item,
            "request_id": uid,
            "video_path": str(video),
            "video_sha256": sha256_file(video),
            "source_duration_s": duration,
            "query": query,
            "prompt_text": prompt(query),
            "protocol_version": core.DEFAULT_INPUT_PROTOCOL,
            "model_revision": core.EXPECTED_MODEL_REVISION,
            "video_contract": dict(VIDEO_CONTRACT),
            "context_token_budget": 16384,
            "decode": {"max_new_tokens": 512},
        })
    if not rows:
        raise ValueError("Input contains no requests")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local release directory or Hugging Face model ID")
    parser.add_argument("--revision", help="Hugging Face revision; use a commit hash to pin weights")
    parser.add_argument("--input", type=Path, required=True, help="JSONL requests; see docs/inference.md")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Confidence cutoff after generation-order NMS (default: retain all survivors)")
    parser.add_argument("--attention-backend", choices=("sdpa", "flash_attention_2"), default="sdpa")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")
    rows = prepare_rows(args.input.resolve(strict=True))
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(
            repo_id=args.model, revision=args.revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*"],
        ))
    model_path = model_path.resolve()
    release = json.loads((model_path / "release.json").read_text())
    if release["format_version"] != 1 or release["base_model_revision"] != core.EXPECTED_MODEL_REVISION:
        raise ValueError("Unsupported release identity")
    head = model_path / release["head"]["weights"]
    head_config = model_path / release["head"]["config"]
    if sha256_file(head_config) != release["head"]["config_sha256"]:
        raise ValueError("Head configuration checksum mismatch")
    runtime_manifest = {
        "status": "complete", "adapter_loaded": False,
        "model_path": str(model_path), "model_files": release["model_files"],
        "step": release["step"],
        "checkpoint_marker_sha256": release["checkpoint_marker_sha256"],
        "online_head": {"updates": release["head"]["updates"],
                        "head_sha256": release["head"]["sha256"],
                        "head_config_sha256": release["head"]["config_sha256"]},
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grounding-confidence-") as temporary:
        work = Path(temporary)
        requests = work / "requests.jsonl"
        requests.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = work / "model-manifest.json"
        manifest.write_text(json.dumps(runtime_manifest, indent=2) + "\n")
        install_into_core(core, sha256_file(Path(core.__file__)))
        arguments = [
            "infer", "--arm", "joint", "--model-path", str(model_path),
            "--model-revision", release["base_model_revision"],
            "--full-model-manifest", str(manifest),
            "--expected-full-model-manifest-sha256", sha256_file(manifest),
            "--span-confidence-head", str(head),
            "--expected-span-confidence-head-sha256", release["head"]["sha256"],
            "--span-head-config", str(head_config),
            "--expected-span-head-config-sha256", release["head"]["config_sha256"],
            "--input", str(requests), "--expected-input-sha256", sha256_file(requests),
            "--expected-rows", str(len(rows)), "--output-dir", str(args.output_dir),
            "--max-new-tokens", "512", "--candidate-aware-stop", "--exact-dedup",
            "--salvage-truncated-interval-prefix", "--attention-backend", args.attention_backend,
        ]
        original_argv = sys.argv
        try:
            sys.argv = arguments
            result = core.main()
        finally:
            sys.argv = original_argv
        shutil.copy2(requests, args.output_dir / "requests.jsonl")
        shutil.copy2(model_path / "release.json", args.output_dir / "release.json")
    with (args.output_dir / "predictions.jsonl").open("w") as output:
        for line in (args.output_dir / "outputs.jsonl").read_text().splitlines():
            row = json.loads(line)
            scores = [p["predicted_span_correctness_probability"] for p in row.get("span_predictions", [])]
            row["span_scores"] = scores
            selected = []
            if row.get("infer_status") == "success":
                candidates = row.get("candidate_intervals", [])
                if len(candidates) != len(scores):
                    raise ValueError("Candidate/score alignment mismatch")
                _, _, survivors, indices = pool(json.dumps(candidates), row["source_duration_s"])
                selected = [span for span, index in zip(survivors, indices) if scores[index] >= args.threshold]
            row["selected_intervals"] = selected
            row["confidence_threshold"] = args.threshold
            output.write(json.dumps(row) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
