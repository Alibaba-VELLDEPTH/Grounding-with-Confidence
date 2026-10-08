#!/usr/bin/env python3
"""Run a matched generator and confidence head, optionally followed by evaluation."""
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
from evaluate import evaluate_rows, read_rows, select_intervals, spans
from head_checkpoint import validate_config
from tvg_confidence.common import sha256_file
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


def prepare_rows(input_path, require_gt=False):
    rows, seen, video_hashes = [], set(), {}
    for item in read_rows(input_path):
        uid = str(item["request_id"])
        if not uid or uid in seen:
            raise ValueError(f"Empty or duplicate request_id: {uid}")
        seen.add(uid)
        video = Path(item["video_path"]).expanduser()
        if not video.is_absolute():
            video = input_path.parent / video
        video = video.resolve(strict=True)
        if not video.is_file():
            raise ValueError(f"Video must be a file: {uid}")
        duration = float(item["source_duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid source_duration_s: {uid}")
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValueError(f"Empty or invalid query: {uid}")
        query = item["query"].strip()
        if require_gt and "gt_intervals" not in item:
            raise ValueError(f"Evaluation requires gt_intervals for every request: {uid}")
        if "gt_intervals" in item:
            spans(item["gt_intervals"], duration, "ground truth")
        if video not in video_hashes:
            video_hashes[video] = sha256_file(video)
        rows.append({
            **item, "request_id": uid, "video_path": str(video),
            "video_sha256": video_hashes[video], "source_duration_s": duration,
            "query": query, "prompt_text": prompt(query),
            "protocol_version": core.DEFAULT_INPUT_PROTOCOL,
            "model_revision": core.EXPECTED_MODEL_REVISION,
            "video_contract": dict(VIDEO_CONTRACT),
            "context_token_budget": 16384, "decode": {"max_new_tokens": 512},
        })
    if not rows:
        raise ValueError("Input contains no requests")
    return rows


def release_file(model_path, name):
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Release manifest must use flat filenames")
    return model_path / name


def model_arguments(model_path, checkpoint, work):
    if (model_path / "adapter_config.json").exists():
        raise ValueError("--model must be the full generator/base, not an auto-loaded adapter")
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve(strict=True)
        adapter = checkpoint / "generator-adapter"
        head = checkpoint / "span-confidence-head.safetensors"
        config = checkpoint / "span-head-config.json"
        adapter_sha = sha256_file(adapter / "adapter_model.safetensors")
        config_sha = sha256_file(adapter / "adapter_config.json")
        head_config = json.loads(config.read_text())
        identity = head_config["generator_identity"]
        if (identity.get("adapter_model_sha256") != adapter_sha
                or identity.get("adapter_config_sha256") != config_sha):
            raise ValueError("SFT generator/head pairing mismatch")
        model_args = ["--adapter-dir", str(adapter),
                      "--expected-adapter-model-sha256", adapter_sha,
                      "--expected-adapter-config-sha256", config_sha]
        metadata = {"kind": "sft_adapter", "base_model_revision": core.EXPECTED_MODEL_REVISION,
                    "adapter_model_sha256": adapter_sha, "adapter_config_sha256": config_sha,
                    "head_sha256": sha256_file(head), "head_config_sha256": sha256_file(config)}
    else:
        metadata = json.loads((model_path / "release.json").read_text())
        if metadata["format_version"] != 1 or metadata["base_model_revision"] != core.EXPECTED_MODEL_REVISION:
            raise ValueError("Unsupported release identity")
        if not metadata["model_files"]:
            raise ValueError("Empty generator manifest")
        expected_weights = {name for name in metadata["model_files"]
                            if name.startswith("model") and name.endswith(".safetensors")}
        if {path.name for path in model_path.glob("model*.safetensors")} != expected_weights:
            raise ValueError("Generator weight inventory mismatch")
        for name, digest in metadata["model_files"].items():
            if sha256_file(release_file(model_path, name)) != digest:
                raise ValueError(f"Generator checksum mismatch: {name}")
        head = release_file(model_path, metadata["head"]["weights"])
        config = release_file(model_path, metadata["head"]["config"])
        if (sha256_file(head) != metadata["head"]["sha256"]
                or sha256_file(config) != metadata["head"]["config_sha256"]):
            raise ValueError("Confidence-head or configuration checksum mismatch")
        runtime_manifest = {
            "status": "complete", "adapter_loaded": False,
            "model_path": str(model_path), "model_files": metadata["model_files"],
            "step": metadata["step"], "checkpoint_marker_sha256": metadata["checkpoint_marker_sha256"],
            "online_head": {"updates": metadata["head"]["updates"],
                            "head_sha256": metadata["head"]["sha256"],
                            "head_config_sha256": metadata["head"]["config_sha256"]},
        }
        validate_config(json.loads(config.read_text()), metadata["head"]["sha256"], runtime_manifest)
        manifest = work / "model-manifest.json"
        manifest.write_text(json.dumps(runtime_manifest, indent=2) + "\n")
        model_args = ["--full-model-manifest", str(manifest),
                      "--expected-full-model-manifest-sha256", sha256_file(manifest)]
    return model_args + [
        "--span-confidence-head", str(head),
        "--expected-span-confidence-head-sha256", sha256_file(head),
        "--span-head-config", str(config),
        "--expected-span-head-config-sha256", sha256_file(config),
    ], metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local release directory or Hugging Face model ID")
    parser.add_argument("--revision", help="Hugging Face revision; pin a commit hash for reproducibility")
    parser.add_argument("--sft-checkpoint", type=Path,
                        help="SFT checkpoint containing generator-adapter and head; --model is the base")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Confidence cutoff after generation-order NMS; default retains all survivors")
    parser.add_argument("--evaluate", action="store_true", help="Require GT and write evaluation.json after inference")
    parser.add_argument("--metric-protocol", choices=("omtg", "set"), default="omtg")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and artifact hashes without loading a GPU model")
    parser.add_argument("--attention-backend", choices=("sdpa", "flash_attention_2"), default="sdpa")
    args = parser.parse_args()
    if args.evaluate and args.metric_protocol == "omtg":
        from scipy.optimize import linear_sum_assignment
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")
    rows = prepare_rows(args.input.resolve(strict=True), require_gt=args.evaluate)
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        if model_path.is_absolute() or args.model.startswith((".", "~")):
            parser.error("local model directory does not exist")
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(
            repo_id=args.model,
            revision=args.revision or (core.EXPECTED_MODEL_REVISION if args.sft_checkpoint else None),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*"],
        ))
    model_path = model_path.resolve()
    with tempfile.TemporaryDirectory(prefix="grounding-confidence-") as temporary:
        work = Path(temporary)
        model_args, metadata = model_arguments(model_path, args.sft_checkpoint, work)
        if args.check_only:
            print(json.dumps({"status": "preflight_passed", "requests": len(rows),
                              "gpu_executed": False, "evaluate": args.evaluate}))
            return 0
        requests = work / "requests.jsonl"
        requests.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        install_into_core(core, sha256_file(Path(core.__file__)))
        arguments = [
            "infer", "--arm", "joint", "--model-path", str(model_path),
            "--model-revision", core.EXPECTED_MODEL_REVISION, *model_args,
            "--input", str(requests), "--expected-input-sha256", sha256_file(requests),
            "--expected-rows", str(len(rows)), "--output-dir", str(args.output_dir),
            "--max-new-tokens", "512", "--candidate-aware-stop", "--exact-dedup",
            "--salvage-truncated-interval-prefix", "--attention-backend", args.attention_backend,
        ]
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        original_argv = sys.argv
        try:
            sys.argv = arguments
            result = core.main()
        finally:
            sys.argv = original_argv
        shutil.copy2(requests, args.output_dir / "requests.jsonl")
        (args.output_dir / "release.json").write_text(json.dumps(metadata, indent=2) + "\n")
    predictions = read_rows(args.output_dir / "outputs.jsonl")
    if (len(predictions) != len(rows)
            or {str(row["request_id"]) for row in predictions} != {row["request_id"] for row in rows}):
        raise ValueError("Incomplete inference outputs; no evaluation was produced")
    with (args.output_dir / "predictions.jsonl").open("x") as output:
        for row in predictions:
            row["span_scores"] = [p["predicted_span_correctness_probability"]
                                  for p in row.get("span_predictions", [])]
            row["selected_intervals"] = select_intervals(row, args.threshold)
            row["confidence_threshold"] = args.threshold
            output.write(json.dumps(row, allow_nan=False) + "\n")
    if args.evaluate:
        report = evaluate_rows(predictions, args.threshold, rows, args.metric_protocol)
        (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
