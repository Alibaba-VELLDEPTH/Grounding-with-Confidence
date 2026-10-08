#!/usr/bin/env python3
"""Prepare v1.3 SFT from caller-owned normalized units and scored replay."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

from sft_targets import PAIR, SUFFIX, PoolBuilder, attach_replay
from train_sft import (
    BOUNDED_POOL_SOFT_CONFIDENCE_V13_PROTOCOL as PROTOCOL,
    LORA_TARGETS,
    VIDEO_CONTRACT,
    load_manifest,
    prepare_records,
)
from tvg_confidence.common import read_jsonl, sha256_file, write_json, write_jsonl

BASE_REVISION = "ddbb6cb944f13ce21e59e85da23c5f356107260e"
POOL_SETTINGS = dict(
    extras_per_gt=2, min_model_tiou=0.5, min_aug_tiou=0.6,
    near_duplicate_tiou=0.85, max_candidates=32, target_tokens=480,
    decode_tokens=512, context_tokens=16384,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: expected nonempty text")
    return value


def intervals(value, label, duration=None, *, allow_empty=False):
    require(isinstance(value, list) and (value or allow_empty), f"{label}: expected intervals")
    for pair in value:
        require(isinstance(pair, list) and len(pair) == 2, f"{label}: invalid interval")
        require(all(type(x) in (int, float) and math.isfinite(x) for x in pair),
                f"{label}: coordinates must be finite numbers")
        require(pair[0] < pair[1], f"{label}: non-positive interval")
        if duration is not None:
            require(0 <= pair[0] < pair[1] <= duration, f"{label}: interval outside duration")
    return value


def training_schedule(rows, world_size=8):
    require(type(rows) is int and rows > 0, "training requires at least one unit")
    require(type(world_size) is int and world_size in (1, 2, 4, 8, 16, 32, 64),
            "world size must be a positive divisor of 64")
    steps = (rows + 63) // 64
    accumulation = (2 * rows + world_size * steps - 1) // (world_size * steps)
    return {
        "target_optimizer_steps_per_epoch": steps,
        "target_optimizer_steps": steps * 2,
        "target_optimizer_steps_per_arm": steps * 2,
        "gradient_accumulation_steps": accumulation,
        "zero_weight_padding_micro_samples_per_epoch": world_size * steps * accumulation - 2 * rows,
    }


def checked_video(meta, root, verified):
    path = Path(text(meta.get("video_path"), "video_path")).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_file(), f"video is not mounted: {path}")
    expected = text(meta.get("video_sha256"), "video_sha256")
    if path not in verified:
        verified[path] = sha256_file(path)
    require(verified[path] == expected, f"video SHA mismatch: {path}")
    meta["video_path"] = str(path)


def validate_unit(unit, root, verified, rounding_tolerance):
    unit = deepcopy(unit)
    uid = text(unit.get("coverage_unit_id"), "coverage_unit_id")
    meta = unit["meta"]
    text(meta.get("query"), f"{uid}/query")
    text(meta.get("source_video_uid"), f"{uid}/source_video_uid")
    duration = meta.get("source_duration_s")
    require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0,
            f"{uid}: invalid source duration")
    require(meta.get("video_contract", VIDEO_CONTRACT) == VIDEO_CONTRACT,
            f"{uid}: video contract mismatch")
    meta["video_contract"] = dict(VIDEO_CONTRACT)
    checked_video(meta, root, verified)
    gt = intervals(unit.get("gt"), f"{uid}/gt", duration)
    original = intervals(unit.get("original_gt"), f"{uid}/original_gt")
    require(gt == sorted(gt) and len({tuple(x) for x in gt}) == len(gt) <= 32,
            f"{uid}: GT must be sorted, unique, and fit the 32-candidate cap")
    require(original == sorted(original) and len(original) == len(gt),
            f"{uid}: original GT alignment mismatch")
    for aligned, raw in zip(gt, original, strict=True):
        clipped = [max(0.0, min(duration, x)) for x in raw]
        require(aligned == clipped and all(abs(a - b) <= rounding_tolerance for a, b in zip(raw, clipped)),
                f"{uid}: GT may only be clipped within the declared rounding tolerance")
    intervals(unit.get("proposals"), f"{uid}/proposals", duration, allow_empty=True)
    source_ids = unit.get("source_ids")
    require(isinstance(source_ids, list) and source_ids, f"{uid}: missing source_ids")
    for source_id in source_ids:
        text(source_id, f"{uid}/source_ids")
    require(len(source_ids) == len(set(source_ids)), f"{uid}: duplicate source IDs")
    return unit


def check_completion(tokenizer, target, budget):
    target_ids = tokenizer.encode(target, add_special_tokens=False).ids
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False).ids
    ids = tokenizer.encode(target + SUFFIX, add_special_tokens=False).ids
    require(ids == target_ids + suffix_ids, "target/suffix tokenizer boundary changed")
    require(0 < len(ids) <= budget, f"completion exceeds {budget}-token budget")
    return len(ids), hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_replay(row, source, tokenizer):
    for key in ("source_video_uid", "video_sha256", "query"):
        require(row[key] == source.get(key), f"replay {key} mismatch: {row['coverage_unit_id']}")
    original = intervals(source.get("gt_intervals"), "replay/gt_intervals")
    require(row["original_gt_intervals"] == sorted(original), "replay GT mismatch")
    target = text(source.get("direct_target_override"), "replay/direct_target_override")
    candidates = intervals(json.loads(target), "replay/candidates", row["source_duration_s"])
    slots = source.get("direct_candidate_slots")
    matches = list(PAIR.finditer(target))
    require(isinstance(slots, list) and len(slots) == len(candidates) == len(matches),
            "replay candidate slots do not align with interval text")
    require([json.loads(match.group()) for match in matches] == candidates,
            "replay character spans do not match candidates")
    ids = [text(slot.get("candidate_id"), "replay/candidate_id") for slot in slots]
    require(len(set(ids)) == len(ids), "duplicate replay candidate ID")
    for slot in slots:
        text(slot.get("slot_role"), "replay/slot_role")
    text(source.get("direct_prompt_text") or source.get("prompt_text"), "replay/prompt_text")
    check_completion(tokenizer, target, 2048)
    return candidates


def validate_teacher(teacher, binding):
    cid = teacher["candidate_id"]
    for key in ("source_video_uid", "query", "candidate_interval_source_s"):
        require(teacher.get(key) == binding[key], f"teacher {key} mismatch: {cid}")
    for key in ("video_sha256", "request_id"):
        if key in teacher:
            require(teacher[key] == binding[key], f"teacher {key} mismatch: {cid}")
    probability = teacher.get("teacher_yes_probability")
    nested = (teacher.get("scores") or {}).get("yes_probability")
    if probability is None:
        probability = nested
    elif nested is not None:
        require(probability == nested, f"conflicting teacher probabilities: {cid}")
    require(type(probability) in (int, float) and math.isfinite(probability) and 0 < probability < 1,
            f"teacher must contain an original continuous P(Yes) in (0, 1): {cid}")
    return probability


def model_inventory(base, tokenizer_path, expected_tokenizer_sha256):
    require(base.is_dir(), f"local pinned base model directory is missing: {base}")
    require(not (base / "adapter_config.json").exists(), "fresh base must not contain an adapter_config.json")
    require(sha256_file(tokenizer_path) == expected_tokenizer_sha256, "tokenizer SHA mismatch")
    require(sha256_file(base / "tokenizer.json") == expected_tokenizer_sha256,
            "preparation tokenizer differs from the base model tokenizer")
    config = json.loads((base / "config.json").read_text())
    require(config.get("model_type") == "qwen3_vl" and config.get("text_config", {}).get("hidden_size") == 2560,
            "v1.3 requires a Qwen3-VL/TimeLens2-4B base with hidden_size=2560")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"}
    index = base / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        require(bool(weight_map), "empty base model weight map")
        required.update(weight_map.values())
        required.add(index.name)
    else:
        required.add("model.safetensors")
    require(any((base / name).is_file() for name in ("chat_template.json", "chat_template.jinja"))
            or bool(json.loads((base / "tokenizer_config.json").read_text()).get("chat_template")),
            "local model is missing its chat template")
    for name in required:
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                f"model weight index must use relative paths: {name}")
        require((base / name).is_file() and (base / name).stat().st_size > 0,
                f"missing or empty base model file: {name}")
    files = required | {str(path.relative_to(base)) for path in base.iterdir()
                        if path.is_file() and path.suffix in (".json", ".jinja")}
    return {name: sha256_file(base / name) for name in sorted(files)}


def materialize(args):
    require(not sys.flags.optimize, "do not use python -O: target-builder assertions are required")
    require(args.model_revision == BASE_REVISION, "v1.3 freezes the original TimeLens2-4B revision")
    require(math.isfinite(args.gt_rounding_tolerance_s) and args.gt_rounding_tolerance_s >= 0,
            "rounding tolerance must be finite and nonnegative")
    output = args.output_dir.expanduser().resolve()
    require(not output.exists(), f"refusing existing output directory: {output}")
    require(output.parent.is_dir(), f"output parent must already exist: {output.parent}")
    paths = {name: getattr(args, name).expanduser().resolve() for name in ("units", "replay", "teacher_scores")}
    lineage = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    base = args.base_model.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    files = model_inventory(base, tokenizer_path, args.expected_tokenizer_sha256)
    builder = PoolBuilder(argparse.Namespace(tokenizer=tokenizer_path, **POOL_SETTINGS))
    verified_videos = {}
    units = [validate_unit(unit, paths["units"].parent, verified_videos, args.gt_rounding_tolerance_s)
             for unit in read_jsonl(paths["units"])]
    schedule = training_schedule(len(units), args.world_size)
    require(len({u["coverage_unit_id"] for u in units}) == len(units), "duplicate coverage_unit_id")
    identities = [(u["meta"]["source_video_uid"], " ".join(u["meta"]["query"].split()).casefold()) for u in units]
    require(len(set(identities)) == len(units), "duplicate normalized video/query unit")
    video_identities = {}
    for unit in units:
        meta = unit["meta"]
        identity = (meta["video_sha256"], meta["source_duration_s"])
        previous = video_identities.setdefault(meta["source_video_uid"], identity)
        require(previous == identity, "inconsistent source video checksum/duration")
    requested = {sorted(unit["source_ids"])[0] for unit in units}
    require(len(requested) == len(units), "selected replay request is shared by multiple units")
    sources = {}
    for source in read_jsonl(paths["replay"]):
        sid = text(source.get("request_id"), "replay/request_id")
        if sid in requested:
            require(sid not in sources, f"duplicate replay request: {sid}")
            sources[sid] = source
    require(set(sources) == requested, f"missing selected replay requests: {len(requested - sources.keys())}")
    rows, bindings = [], {}
    used_teachers, roles = {}, Counter()
    for unit in units:
        uid = unit["coverage_unit_id"]
        row = builder.row(uid, unit, "supported_expansion")
        row.update(protocol_version=PROTOCOL, request_id=PROTOCOL + "_" + uid)
        source = sources[sorted(unit["source_ids"])[0]]
        candidates = validate_replay(row, source, builder.tokenizer)
        attach_replay(row, source, builder.tokenizer, used_teachers, roles)
        for slot, candidate in zip(source["direct_candidate_slots"], candidates, strict=True):
            binding = {key: row[key] for key in ("source_video_uid", "video_sha256", "query")}
            binding.update(request_id=source["request_id"], candidate_interval_source_s=candidate)
            cid = slot["candidate_id"]
            require(bindings.setdefault(cid, binding) == binding, f"reused candidate ID has inconsistent identity: {cid}")
        rows.append(row)
    teachers, seen = [], set()
    for teacher in read_jsonl(paths["teacher_scores"]):
        cid = text(teacher.get("candidate_id"), "teacher/candidate_id")
        require(cid not in seen, f"duplicate teacher candidate ID: {cid}")
        seen.add(cid)
        if cid in bindings:
            validate_teacher(teacher, bindings[cid])
            teachers.append(teacher)
    require(set(bindings) <= seen, f"missing teacher scores: {len(set(bindings) - seen)}")
    for name, path in paths.items():
        require(sha256_file(path) == lineage[name]["sha256"], f"source changed during preparation: {path}")

    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        write_jsonl(staging / "input.jsonl", rows)
        write_jsonl(staging / "teacher-scores.jsonl", teachers)
        manifest = {
            "schema_version": 1, "protocol_version": PROTOCOL, "authorized": True,
            "status": "materialized_not_trained", "experiment_name": "v1.3-recipe-caller-data",
            "input": {"path": str(staging / "input.jsonl"), "rows": len(rows),
                      "sha256": sha256_file(staging / "input.jsonl")},
            "model": {"path": str(base), "revision": BASE_REVISION, "files_sha256": files},
            "generator_initialization": None,
            "base_initialization_contract": {"base_model": "TimeLens2-4B", "revision": BASE_REVISION,
                "fresh_lora": True, "previous_adapter_loaded": False, "previous_optimizer_loaded": False},
            "lora": {"rank": 32, "alpha": 64, "dropout": 0.05, "targets": list(LORA_TARGETS)},
            "confidence_supervision": {"kind": "inline_verifier_pyes", "target": "verifier_v2_yes_probability",
                "teacher_path": str(staging / "teacher-scores.jsonl"),
                "teacher_sha256": sha256_file(staging / "teacher-scores.jsonl"),
                "expected_teacher_candidates": len(teachers), "rollout_lm_loss": False,
                "hard_confidence_labels": False, "synthetic_candidate_scores_fabricated": False},
            "rollout_shards": [], "expected_rollout_queries": 0,
            "unique_source_videos": len(video_identities),
            "training_contract": {"run_mode": "joint-only", "arms_sequential": ["joint"],
                "standalone_joint_full": True, "target_global_batch_queries": 64, "epochs_per_arm": 2,
                **schedule, "world_size": args.world_size, "local_world_size": args.world_size,
                "generator_update_mode": "lm_only_with_detached_confidence",
                "confidence_backprop_to_generator": False, "confidence_supervision_mode": "detached_replay_target",
                "lambda_conf": 0.25, "confidence_ranking_weight": 0.25, "confidence_ranking_min_gap": 0.1,
                "generator_lr": 2e-5, "head_lr": 2e-4, "negative_interval_lm_loss": False,
                "objective": "exact GT anchors + bounded supported alternatives; group-normalized weighted CE",
                "loss_group_mass_gt_alternative_format": [0.6, 0.3, 0.1],
                "loss_group_mass_without_alternatives": [0.9, 0, 0.1],
                "confidence_feature": "mean interval text-token hidden, detached",
                "confidence_loss": "soft BCEWithLogits + 0.25 pairwise ranking; teacher probability gap >= 0.1",
                "endpoint_anchor": "after_end", "checkpoint_steps": 500, "checkpoint_total_limit": 4,
                "checkpoint_save_final": True, "training_diagnostics": {"every_steps": 10}},
            "head_initialization": {"kind": "random_span_hidden_binary_mlp", "seed": 20260902,
                "architecture": {"input_dim": 2560, "hidden_dim": 256, "output_dim": 1, "dropout": 0.1}},
            "tokenizer_contract": {"path": str(tokenizer_path), "sha256": args.expected_tokenizer_sha256},
            "runtime_completion_token_cap": 512, "runtime_context_token_cap": 16384,
            "source_lineage": lineage, "pool_settings": POOL_SETTINGS,
            "gt_rounding_tolerance_s": args.gt_rounding_tolerance_s,
            "replay_selection": "sorted(source_ids)[0]; one unchanged caller-supplied replay per unit",
            "paper_main_eligible": False,
        }
        _, summary = prepare_records(manifest)
        manifest["input"]["path"] = str(output / "input.jsonl")
        manifest["confidence_supervision"]["teacher_path"] = str(output / "teacher-scores.jsonl")
        write_json(staging / "input-manifest.json", manifest)
        manifest_sha = sha256_file(staging / "input-manifest.json")
        load_manifest(staging / "input-manifest.json", manifest_sha)
        (staging / "input-manifest.sha256").write_text(manifest_sha + "  input-manifest.json\n")
        report = {"manifest_path": str(output / "input-manifest.json"), "manifest_sha256": manifest_sha,
                  "data": summary, "schedule": schedule, "paper_main_eligible": False,
                  "limitations": "Caller-owned data; original paper training units and verifier files are not distributed. CPU validation does not decode video or run the model."}
        write_json(staging / "preparation-report.json", report)
        output.mkdir()
        for path in staging.iterdir():
            path.rename(output / path.name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True, help="Normalized units JSONL; see training/SFT.md")
    parser.add_argument("--replay", type=Path, required=True, help="Caller-owned offline candidate replay JSONL")
    parser.add_argument("--teacher-scores", type=Path, required=True, help="Original soft verifier scores JSONL; never inferred from GT")
    parser.add_argument("--base-model", type=Path, required=True, help="Local complete pinned TimeLens2-4B snapshot")
    parser.add_argument("--model-revision", choices=[BASE_REVISION], default=BASE_REVISION)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory under an existing parent")
    parser.add_argument("--world-size", type=int, choices=[1, 2, 4, 8, 16, 32, 64], default=8)
    parser.add_argument("--gt-rounding-tolerance-s", type=float, default=0.0)
    args = parser.parse_args()
    try:
        report = materialize(args)
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        raise SystemExit(f"SFT preparation refused: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
