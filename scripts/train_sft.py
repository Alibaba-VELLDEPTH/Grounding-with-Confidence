#!/usr/bin/env python3
"""Manifest-driven SFT with GT-anchored targets and detached confidence replay.

The paper configuration uses a span-pooled LayerNorm-MLP head, verifier
soft targets, BCE plus pairwise ranking, and separate generator/head
gradients. Historical optional training modes remain in this source;
configs/paper_recipe.json identifies the reported configuration.
Training data, teacher scores, model weights, and manifests are external.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ENTRY_PATH = Path(__file__).resolve()
ROOT = next(
    (
        candidate
        for candidate in (ENTRY_PATH.parent, ENTRY_PATH.parents[1])
        if (candidate / "src" / "tvg_confidence").is_dir()
    ),
    ENTRY_PATH.parents[1],
)
sys.path.insert(0, str(ROOT / "src"))
VENDOR_PATH = ENTRY_PATH.parent / "vendor"
if VENDOR_PATH.is_dir():
    sys.path.insert(0, str(VENDOR_PATH))

from tvg_confidence.common import read_jsonl, sha256_file, write_json  # noqa: E402
from tvg_confidence.intervals import canonicalize_interval_set_to_duration  # noqa: E402
from tvg_confidence.semantic import map_token_semantics  # noqa: E402


PROTOCOL = "timelens2_single_stage_generator_joint_sft_v1"
SUPPORTED_PROTOCOLS = {
    PROTOCOL,
    "timelens2_single_stage_generator_joint_sft_after_end_v2",
    "timelens2_single_stage_generator_joint_sft_after_end_distributed_full_v1",
    "timelens2_single_stage_generator_joint_sft_highcard_distributed_v1",
    "timelens2_single_stage_generator_joint_sft_prompt_augmented_h20_v1",
    "timelens2_explicit_candidate_confidence_sft_h20_v1",
    "timelens2_strong_r7_verifier_pyes_endpoint_sft_v1",
    "timelens2_residual_completion_pyes_joint_sft_v1",
    "timelens2_expanded_gt_sft_control_v1",
    "timelens2_revised_multispan_gt_only_sft_control_v1",
    "timelens2_expanded_residual_completion_sft_v1",
    "timelens2_positive_only_set_coverage_sft_v1",
    "timelens2_positive_only_early_close_sft_v2",
    "timelens2_positive_only_exact_duplicate_penalty_sft_v3",
    "timelens2_positive_only_fixed64_candidates_sft_v4",
    "timelens2_positive_only_fixed16_hungarian_confidence_sft_v6",
    "timelens2_positive_only_fixed32_all_gt_random_negative_confidence_sft_v7",
    "timelens2_positive_only_fixed32_gt_nms_anchor_coordinate_sft_v8",
    "timelens2_positive_only_variable_close_detached_confidence_sft_v1_1",
    "timelens2_bounded_coverage_pool_sft_v1",
    "timelens2_bounded_coverage_pool_soft_confidence_sft_v1",
    "timelens2_bounded_coverage_pool_soft_confidence_sft_v1_2",
    "timelens2_bounded_coverage_pool_soft_confidence_sft_v1_3",
    "timelens2_bounded_pool_gt_binary_sft_v1",
}
ROLLOUT_PROTOCOL = "timelens2_single_stage_generator_count9_rollout_v1"
HIGHCARD_DIRECT_PROTOCOL = (
    "timelens2_single_stage_generator_joint_sft_highcard_distributed_v1"
)
PROMPT_AUGMENTED_PROTOCOL = (
    "timelens2_single_stage_generator_joint_sft_prompt_augmented_h20_v1"
)
EXPLICIT_CANDIDATE_CONFIDENCE_PROTOCOL = (
    "timelens2_explicit_candidate_confidence_sft_h20_v1"
)
RESIDUAL_COMPLETION_PYES_PROTOCOL = (
    "timelens2_residual_completion_pyes_joint_sft_v1"
)
EXPANDED_GT_SFT_CONTROL_PROTOCOL = "timelens2_expanded_gt_sft_control_v1"
REVISED_MULTISPAN_GT_ONLY_PROTOCOL = (
    "timelens2_revised_multispan_gt_only_sft_control_v1"
)
EXPANDED_RESIDUAL_COMPLETION_PROTOCOL = (
    "timelens2_expanded_residual_completion_sft_v1"
)
POSITIVE_ONLY_SET_COVERAGE_PROTOCOL = (
    "timelens2_positive_only_set_coverage_sft_v1"
)
POSITIVE_ONLY_EARLY_CLOSE_PROTOCOL = (
    "timelens2_positive_only_early_close_sft_v2"
)
POSITIVE_ONLY_EXACT_DUPLICATE_PROTOCOL = (
    "timelens2_positive_only_exact_duplicate_penalty_sft_v3"
)
POSITIVE_ONLY_FIXED64_PROTOCOL = (
    "timelens2_positive_only_fixed64_candidates_sft_v4"
)
POSITIVE_ONLY_FIXED16_HUNGARIAN_CONFIDENCE_PROTOCOL = (
    "timelens2_positive_only_fixed16_hungarian_confidence_sft_v6"
)
POSITIVE_ONLY_FIXED32_ALL_GT_RANDOM_NEGATIVE_CONFIDENCE_PROTOCOL = (
    "timelens2_positive_only_fixed32_all_gt_random_negative_confidence_sft_v7"
)
POSITIVE_ONLY_FIXED32_GT_NMS_ANCHOR_COORDINATE_PROTOCOL = (
    "timelens2_positive_only_fixed32_gt_nms_anchor_coordinate_sft_v8"
)
POSITIVE_ONLY_VARIABLE_CLOSE_DETACHED_CONFIDENCE_PROTOCOL = (
    "timelens2_positive_only_variable_close_detached_confidence_sft_v1_1"
)
BOUNDED_COVERAGE_POOL_PROTOCOL = "timelens2_bounded_coverage_pool_sft_v1"
BOUNDED_POOL_SOFT_CONFIDENCE_PROTOCOL = "timelens2_bounded_coverage_pool_soft_confidence_sft_v1"
BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOL = "timelens2_bounded_coverage_pool_soft_confidence_sft_v1_2"
BOUNDED_POOL_SOFT_CONFIDENCE_V13_PROTOCOL = "timelens2_bounded_coverage_pool_soft_confidence_sft_v1_3"
BOUNDED_POOL_GT_BINARY_PROTOCOL = "timelens2_bounded_pool_gt_binary_sft_v1"
BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS = {
    BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOL,
    BOUNDED_POOL_SOFT_CONFIDENCE_V13_PROTOCOL,
    BOUNDED_POOL_GT_BINARY_PROTOCOL,
}
EXPECTED_QUERIES = 6_500
MIN_CONFIDENCE_ENDPOINTS = 5_000
THRESHOLDS = (0.3, 0.5, 0.7)
# Explicit-confidence targets can contain more than one hundred candidates.
# Materializing FP32 logits for every supervised token at once costs close to
# one GiB for Qwen3-VL's vocabulary on the longest OMTG examples.  Checkpoint
# small LM-head chunks so backward recomputes their logits instead of retaining
# the full target-by-vocabulary tensor.
LM_HEAD_CHECKPOINT_CHUNK_TOKENS = 32
VIDEO_CONTRACT = {
    "fps": 2.0,
    "max_frames": 120,
    "min_pixels": 4096,
    "max_pixels": 65536,
    "total_pixels": 7_864_320,
}
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def direct_tiou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(
        0.0,
        min(float(left[1]), float(right[1]))
        - max(float(left[0]), float(right[0])),
    )
    union = max(float(left[1]), float(right[1])) - min(
        float(left[0]), float(right[0])
    )
    return intersection / union if union > 0.0 else 0.0


def direct_prompt(query: str) -> str:
    return (
        "You are given a video.\n"
        "Task: temporal retrieval.\n"
        f'Find every distinct time span in which the query "{query}" occurs.\n'
        "Return all matching non-overlapping time spans in seconds. Do not merge "
        "separate occurrences and do not invent an occurrence.\n"
        "Output format MUST be a JSON array of [start, end] pairs, for example "
        "[[0,3.5],[10,12]].\n"
    )


def normalized_gt_target(intervals: Sequence[Sequence[float]]) -> str:
    normalized = [
        [float(start), float(end)]
        for start, end in intervals
    ]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def checked_lm_masked_char_spans(
    value: Any, target: str, *, request_id: str
) -> list[list[int]]:
    """Validate half-open target character spans excluded from LM loss."""

    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise SystemExit(f"LM masked spans are not a list: {request_id}")
    normalized: list[list[int]] = []
    previous_end = 0
    for span in value:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or type(span[0]) is not int
            or type(span[1]) is not int
        ):
            raise SystemExit(f"invalid LM masked character span: {request_id}/{span!r}")
        start, end = span
        if start < previous_end or start < 0 or end <= start or end > len(target):
            raise SystemExit(
                f"out-of-range/overlapping LM masked span: {request_id}/{span!r}"
            )
        normalized.append([start, end])
        previous_end = end
    return normalized


def checked_lm_weighted_char_spans(
    value: Any, target: str, *, request_id: str
) -> list[dict[str, Any]]:
    """Validate target character spans carrying an explicit LM-token weight."""

    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise SystemExit(f"LM weighted spans are not a list: {request_id}")
    normalized: list[dict[str, Any]] = []
    previous_end = 0
    for item in value:
        if not isinstance(item, dict):
            raise SystemExit(f"invalid LM weighted span: {request_id}/{item!r}")
        span = item.get("char_span")
        weight = item.get("weight")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or type(span[0]) is not int
            or type(span[1]) is not int
            or not isinstance(weight, (int, float))
        ):
            raise SystemExit(f"invalid LM weighted span: {request_id}/{item!r}")
        start, end = span
        numeric_weight = float(weight)
        if (
            start < previous_end
            or start < 0
            or end <= start
            or end > len(target)
            or not math.isfinite(numeric_weight)
            or numeric_weight <= 0.0
        ):
            raise SystemExit(f"out-of-range LM weighted span: {request_id}/{item!r}")
        normalized.append(
            {
                "char_span": [start, end],
                "weight": numeric_weight,
                "kind": str(item.get("kind") or "unspecified"),
            }
        )
        previous_end = end
    return normalized


def load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise SystemExit("training manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    from tvg_confidence.gt_binary_supervision import validate_contract
    validate_contract(manifest)
    if manifest.get("protocol_version") not in SUPPORTED_PROTOCOLS:
        raise SystemExit("training manifest protocol mismatch")
    if manifest.get("authorized") is not True:
        raise SystemExit("training manifest is not authorized")
    return manifest


def checked_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise SystemExit(f"data SHA mismatch: {path}")
    return list(read_jsonl(path))


def prepare_records(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_spec = manifest["input"]
    expected_queries = int(input_spec["rows"])
    expected_rollout_queries = int(
        manifest.get("expected_rollout_queries", EXPECTED_QUERIES)
    )
    input_rows = checked_rows(
        Path(input_spec["path"]), str(input_spec["sha256"])
    )
    if manifest.get("protocol_version") in {
        EXPLICIT_CANDIDATE_CONFIDENCE_PROTOCOL,
    }:
        first_missing_video = next(
            (
                row
                for row in input_rows
                if not Path(str(row["video_path"])).exists()
            ),
            None,
        )
        if first_missing_video is not None:
            raise SystemExit(
                "explicit-confidence training video is not mounted: "
                f"{first_missing_video['video_path']}"
            )
    output_rows: list[dict[str, Any]] = []
    for shard in manifest.get("rollout_shards", []):
        rows = checked_rows(Path(shard["output_path"]), str(shard["output_sha256"]))
        run_manifest_path = Path(shard["run_manifest_path"])
        if sha256_file(run_manifest_path) != str(shard["run_manifest_sha256"]):
            raise SystemExit(f"run-manifest SHA mismatch: {run_manifest_path}")
        run_manifest = json.loads(run_manifest_path.read_text())
        completion = run_manifest.get("completion", {})
        expected_shard_rows = int(shard.get("rows", 1_625))
        if (
            run_manifest.get("status") != "success"
            or int(completion.get("expected_rows", -1)) != expected_shard_rows
            or int(completion.get("success_rows", -1)) != expected_shard_rows
            or int(completion.get("failed_rows", -1)) != 0
        ):
            raise SystemExit(f"rollout shard not complete: {run_manifest_path}")
        output_rows.extend(rows)

    input_by_id = {str(row["request_id"]): row for row in input_rows}
    output_by_id = {str(row["request_id"]): row for row in output_rows}
    identity_ok = (
        len(input_rows) == len(input_by_id) == expected_queries
        and len(output_rows) == len(output_by_id) == expected_rollout_queries
        and set(output_by_id).issubset(input_by_id)
    )
    if not identity_ok:
        raise SystemExit("input/output request identity mismatch")

    confidence_spec = manifest.get("confidence_supervision") or {
        "kind": "max_direct_tiou"
    }
    confidence_kind = str(confidence_spec.get("kind"))
    teacher_by_request: dict[str, list[dict[str, Any]]] = {}
    teacher_by_candidate: dict[str, dict[str, Any]] = {}
    if confidence_kind == "verifier_pyes":
        teacher_path = Path(str(confidence_spec["teacher_path"]))
        if sha256_file(teacher_path) != str(confidence_spec["teacher_sha256"]):
            raise SystemExit("verifier P(Yes) teacher SHA mismatch")
        for teacher_row in read_jsonl(teacher_path):
            teacher_by_request.setdefault(
                str(teacher_row["request_id"]), []
            ).append(teacher_row)
        if (
            len(teacher_by_request)
            != int(confidence_spec["expected_teacher_queries"])
            or sum(map(len, teacher_by_request.values()))
            != int(confidence_spec["expected_teacher_candidates"])
        ):
            raise SystemExit("verifier P(Yes) teacher volume mismatch")
    elif confidence_kind == "inline_verifier_pyes":
        teacher_path = Path(str(confidence_spec["teacher_path"]))
        if sha256_file(teacher_path) != str(confidence_spec["teacher_sha256"]):
            raise SystemExit("inline verifier P(Yes) teacher SHA mismatch")
        for teacher_row in read_jsonl(teacher_path):
            candidate_id = str(teacher_row["candidate_id"])
            if candidate_id in teacher_by_candidate:
                raise SystemExit(
                    f"duplicate inline verifier candidate ID: {candidate_id}"
                )
            teacher_by_candidate[candidate_id] = teacher_row
        if len(teacher_by_candidate) != int(
            confidence_spec["expected_teacher_candidates"]
        ):
            raise SystemExit("inline verifier P(Yes) teacher volume mismatch")
    elif confidence_kind not in {"max_direct_tiou", "none", "inline_gt_binary"}:
        raise SystemExit(f"unsupported confidence supervision: {confidence_kind}")

    records: list[dict[str, Any]] = []
    canonical_valid = duration_filtered = positive = negative = 0
    for input_position, source in enumerate(input_rows):
        request_id = str(source["request_id"])
        rollout = output_by_id.get(request_id)
        if rollout is None:
            inline_confidence = (
                source.get("protocol_version") == POSITIVE_ONLY_SET_COVERAGE_PROTOCOL
                and confidence_kind == "inline_verifier_pyes"
                and source.get("confidence_supervision_available") is True
            )
            detached_confidence_replay = (
                source.get("protocol_version")
                in {
                    POSITIVE_ONLY_VARIABLE_CLOSE_DETACHED_CONFIDENCE_PROTOCOL,
                    BOUNDED_POOL_SOFT_CONFIDENCE_PROTOCOL,
                    BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOL,
                    BOUNDED_POOL_SOFT_CONFIDENCE_V13_PROTOCOL,
                    BOUNDED_POOL_GT_BINARY_PROTOCOL,
                }
                and confidence_kind in {"inline_verifier_pyes", "inline_gt_binary"}
                and source.get("confidence_supervision_available") is True
                and source.get("detached_confidence_replay_available") is True
            )
            if not (
                source.get("protocol_version")
                in {
                    HIGHCARD_DIRECT_PROTOCOL,
                    PROMPT_AUGMENTED_PROTOCOL,
                    EXPLICIT_CANDIDATE_CONFIDENCE_PROTOCOL,
                    RESIDUAL_COMPLETION_PYES_PROTOCOL,
                    EXPANDED_GT_SFT_CONTROL_PROTOCOL,
                    REVISED_MULTISPAN_GT_ONLY_PROTOCOL,
                    EXPANDED_RESIDUAL_COMPLETION_PROTOCOL,
                    POSITIVE_ONLY_SET_COVERAGE_PROTOCOL,
                    POSITIVE_ONLY_EARLY_CLOSE_PROTOCOL,
                    POSITIVE_ONLY_EXACT_DUPLICATE_PROTOCOL,
                    POSITIVE_ONLY_FIXED64_PROTOCOL,
                    POSITIVE_ONLY_FIXED16_HUNGARIAN_CONFIDENCE_PROTOCOL,
                    POSITIVE_ONLY_FIXED32_ALL_GT_RANDOM_NEGATIVE_CONFIDENCE_PROTOCOL,
                    POSITIVE_ONLY_FIXED32_GT_NMS_ANCHOR_COORDINATE_PROTOCOL,
                    POSITIVE_ONLY_VARIABLE_CLOSE_DETACHED_CONFIDENCE_PROTOCOL,
                    BOUNDED_COVERAGE_POOL_PROTOCOL,
                    BOUNDED_POOL_SOFT_CONFIDENCE_PROTOCOL,
                    BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOL,
                    BOUNDED_POOL_SOFT_CONFIDENCE_V13_PROTOCOL,
                    BOUNDED_POOL_GT_BINARY_PROTOCOL,
                }
                and (
                    source.get("confidence_supervision_available") is False
                    or inline_confidence
                    or detached_confidence_replay
                )
                and source.get("video_contract") == VIDEO_CONTRACT
            ):
                raise SystemExit(f"missing rollout without direct-only contract: {request_id}")
            direct_target = str(
                source.get("direct_target_override")
                or source.get("explicit_confidence_target")
                or normalized_gt_target(source.get("gt_intervals") or [])
            )
            direct_lm_masked_char_spans = checked_lm_masked_char_spans(
                source.get("direct_lm_masked_char_spans"),
                direct_target,
                request_id=request_id,
            )
            direct_lm_weighted_char_spans = checked_lm_weighted_char_spans(
                source.get("direct_lm_weighted_char_spans"),
                direct_target,
                request_id=f"{request_id}/weighted-lm",
            )
            direct_duplicate_unlikelihood_char_spans = checked_lm_masked_char_spans(
                source.get("direct_duplicate_unlikelihood_char_spans"),
                direct_target,
                request_id=f"{request_id}/exact-duplicate",
            )
            if any(
                span not in direct_lm_masked_char_spans
                for span in direct_duplicate_unlikelihood_char_spans
            ):
                raise SystemExit(
                    "exact-duplicate unlikelihood spans must remain masked from LM CE: "
                    f"{request_id}"
                )
            direct_endpoint_targets: list[dict[str, Any]] = []
            if inline_confidence:
                slots = source.get("direct_candidate_slots")
                if not isinstance(slots, list) or not slots:
                    raise SystemExit(
                        f"inline confidence row has no candidate slots: {request_id}"
                    )
                for span_index, slot in enumerate(slots):
                    candidate_id = str(slot["candidate_id"])
                    teacher_row = teacher_by_candidate.get(candidate_id)
                    if teacher_row is None:
                        raise SystemExit(
                            f"inline verifier score is missing: {request_id}/{candidate_id}"
                        )
                    quality_value = teacher_row.get("teacher_yes_probability")
                    if quality_value is None:
                        quality_value = (teacher_row.get("scores") or {}).get(
                            "yes_probability"
                        )
                    quality = float(quality_value)
                    if not 0.0 <= quality <= 1.0:
                        raise SystemExit(
                            f"inline verifier score is out of range: {candidate_id}"
                        )
                    end_char_span = checked_lm_masked_char_spans(
                        [slot["end_char_span"]],
                        direct_target,
                        request_id=f"{request_id}/{candidate_id}/endpoint",
                    )[0]
                    direct_endpoint_targets.append(
                        {
                            "span_index": span_index,
                            "candidate_id": candidate_id,
                            "end_char_span": end_char_span,
                            "max_direct_tiou": quality,
                            "ordinal": [
                                quality >= threshold for threshold in THRESHOLDS
                            ],
                        }
                    )
                    positive += int(quality >= 0.5)
                    negative += int(quality < 0.5)
            confidence_replay_target = ""
            confidence_replay_targets: list[dict[str, Any]] = []
            if detached_confidence_replay:
                confidence_replay_target = str(
                    source.get("detached_confidence_replay_target") or ""
                )
                slots = source.get("detached_confidence_replay_candidate_slots")
                if not confidence_replay_target or not isinstance(slots, list) or not slots:
                    raise SystemExit(
                        f"detached confidence replay is incomplete: {request_id}"
                    )
                binary_targets = None
                if confidence_kind == "inline_gt_binary":
                    from tvg_confidence.gt_binary_supervision import replay_binary_targets
                    binary_targets = replay_binary_targets(source)
                for span_index, slot in enumerate(slots):
                    candidate_id = str(slot["candidate_id"])
                    if binary_targets is not None:
                        quality = binary_targets[span_index]
                    else:
                        teacher_row = teacher_by_candidate.get(candidate_id)
                        if teacher_row is None:
                            raise SystemExit(f"detached verifier score is missing: {request_id}/{candidate_id}")
                        quality_value = teacher_row.get("teacher_yes_probability")
                        if quality_value is None:
                            quality_value = (teacher_row.get("scores") or {}).get("yes_probability")
                        quality = float(quality_value)
                        if not 0.0 <= quality <= 1.0:
                            raise SystemExit(f"detached verifier score is out of range: {candidate_id}")
                    span_char_span = checked_lm_masked_char_spans(
                        [slot["span_char_span"]],
                        confidence_replay_target,
                        request_id=f"{request_id}/{candidate_id}/span",
                    )[0]
                    confidence_replay_targets.append(
                        {
                            "span_index": span_index,
                            "candidate_id": candidate_id,
                            "span_char_span": span_char_span,
                            "max_direct_tiou": quality,
                        }
                    )
                    positive += int(quality >= 0.5)
                    negative += int(quality < 0.5)
            records.append(
                {
                    "input_position": input_position,
                    "request_id": request_id,
                    "source_video_uid": str(source["source_video_uid"]),
                    "video_path": str(source["video_path"]),
                    "video_sha256": str(source["video_sha256"]),
                    "source_duration_s": float(source["source_duration_s"]),
                    "query": str(source["query"]),
                    "gt_intervals": source.get("gt_intervals") or [],
                    "direct_prompt": str(
                        source.get("direct_prompt_text")
                        or direct_prompt(str(source["query"]))
                    ),
                    "direct_target": direct_target,
                    "completion_token_budget": int(source.get("completion_token_budget", 0)),
                    "context_token_budget": int(source.get("context_token_budget", 0)),
                    "expected_completion_tokens": int(source.get("expected_completion_tokens", 0)),
                    "expected_completion_sha256": str(source.get("expected_completion_sha256") or ""),
                    "replay_completion_token_budget": int(source.get("replay_completion_token_budget", 0)),
                    "expected_replay_tokens": int(source.get("expected_replay_tokens", 0)),
                    "expected_replay_sha256": str(source.get("expected_replay_sha256") or ""),
                    "direct_loss_prefix_text": str(
                        source.get("direct_loss_prefix_text") or ""
                    ),
                    "direct_lm_masked_char_spans": direct_lm_masked_char_spans,
                    "direct_lm_weighted_char_spans": direct_lm_weighted_char_spans,
                    "direct_duplicate_unlikelihood_char_spans": (
                        direct_duplicate_unlikelihood_char_spans
                    ),
                    "direct_endpoint_targets": direct_endpoint_targets,
                    "rollout_prompt": str(
                        source.get("detached_confidence_replay_prompt_text")
                        or source.get("direct_prompt_text")
                        or direct_prompt(str(source["query"]))
                    ),
                    "rollout_response": confidence_replay_target,
                    "rollout_completion_ids": [],
                    "rollout_strict_parse_valid": False,
                    "rollout_canonical_valid": False,
                    "rollout_confidence_available": detached_confidence_replay,
                    "rollout_normalization_status": (
                        "detached_teacher_forced_candidate_replay"
                        if detached_confidence_replay
                        else "direct_only_no_rollout"
                    ),
                    "rollout_targets": confidence_replay_targets,
                    "confidence_replay_target": confidence_replay_target,
                    "processor_contract": None,
                }
            )
            continue
        if rollout.get("infer_status") != "success":
            raise SystemExit(f"failed rollout row: {request_id}")
        if (
            source.get("protocol_version")
            not in {
                ROLLOUT_PROTOCOL,
                EXPLICIT_CANDIDATE_CONFIDENCE_PROTOCOL,
                RESIDUAL_COMPLETION_PYES_PROTOCOL,
                EXPANDED_GT_SFT_CONTROL_PROTOCOL,
                REVISED_MULTISPAN_GT_ONLY_PROTOCOL,
                EXPANDED_RESIDUAL_COMPLETION_PROTOCOL,
                POSITIVE_ONLY_SET_COVERAGE_PROTOCOL,
                POSITIVE_ONLY_EARLY_CLOSE_PROTOCOL,
                POSITIVE_ONLY_EXACT_DUPLICATE_PROTOCOL,
                POSITIVE_ONLY_FIXED64_PROTOCOL,
                POSITIVE_ONLY_FIXED16_HUNGARIAN_CONFIDENCE_PROTOCOL,
                POSITIVE_ONLY_FIXED32_ALL_GT_RANDOM_NEGATIVE_CONFIDENCE_PROTOCOL,
                POSITIVE_ONLY_FIXED32_GT_NMS_ANCHOR_COORDINATE_PROTOCOL,
            }
            or source.get("video_contract") != VIDEO_CONTRACT
            or rollout.get("video_contract") != VIDEO_CONTRACT
            or rollout.get("processor_contract", {}).get("requested")
            != VIDEO_CONTRACT
        ):
            raise SystemExit(f"video/protocol contract mismatch: {request_id}")
        response = str(rollout.get("response", ""))
        canonical = canonicalize_interval_set_to_duration(
            response, duration_s=float(source["source_duration_s"])
        )
        parsed = canonical.source_parse
        completion_ids = [int(value) for value in rollout.get("completion_token_ids", [])]
        targets: list[dict[str, Any]] = []
        confidence_available = bool(canonical.parse_valid and canonical.intervals)
        if confidence_available:
            canonical_valid += 1
            duration_filtered += int(canonical.normalization_applied)
            gt = source.get("gt_intervals") or []
            teacher_queues: dict[tuple[float, float], list[dict[str, Any]]] = {}
            if confidence_kind == "verifier_pyes":
                for teacher_row in teacher_by_request.get(request_id, []):
                    key = tuple(
                        map(float, teacher_row["candidate_interval_source_s"])
                    )
                    teacher_queues.setdefault(key, []).append(teacher_row)
                for queue in teacher_queues.values():
                    queue.sort(key=lambda row: int(row["candidate_index"]))
            for span_index, (source_span_index, candidate) in enumerate(
                zip(canonical.source_span_indices, canonical.intervals)
            ):
                if confidence_kind == "verifier_pyes":
                    queue = teacher_queues.get(tuple(map(float, candidate)), [])
                    if not queue:
                        # The teacher materializer intentionally omits candidates
                        # whose duration-normalized clip is too short.  They must
                        # not receive a fabricated confidence target.
                        continue
                    teacher_row = queue.pop(0)
                    quality = float(teacher_row["teacher_yes_probability"])
                else:
                    quality = max(
                        (direct_tiou(candidate, target) for target in gt),
                        default=0.0,
                    )
                targets.append(
                    {
                        "span_index": span_index,
                        "source_span_index": source_span_index,
                        "candidate_interval_source_s": list(map(float, candidate)),
                        "max_direct_tiou": quality,
                        "ordinal": [quality >= threshold for threshold in THRESHOLDS],
                    }
                )
                positive += int(quality >= 0.5)
                negative += int(quality < 0.5)
            if confidence_kind == "verifier_pyes" and any(teacher_queues.values()):
                raise SystemExit(
                    "verifier P(Yes) teacher candidates were not consumed: "
                    f"{request_id}/remaining="
                    f"{sum(map(len, teacher_queues.values()))}"
                )
        direct_target = str(
            source.get("direct_target_override")
            or source.get("explicit_confidence_target")
            or normalized_gt_target(source.get("gt_intervals") or [])
        )
        direct_lm_masked_char_spans = checked_lm_masked_char_spans(
            source.get("direct_lm_masked_char_spans"),
            direct_target,
            request_id=request_id,
        )
        direct_lm_weighted_char_spans = checked_lm_weighted_char_spans(
            source.get("direct_lm_weighted_char_spans"),
            direct_target,
            request_id=f"{request_id}/weighted-lm",
        )
        direct_duplicate_unlikelihood_char_spans = checked_lm_masked_char_spans(
            source.get("direct_duplicate_unlikelihood_char_spans"),
            direct_target,
            request_id=f"{request_id}/exact-duplicate",
        )
        if any(
            span not in direct_lm_masked_char_spans
            for span in direct_duplicate_unlikelihood_char_spans
        ):
            raise SystemExit(
                "exact-duplicate unlikelihood spans must remain masked from LM CE: "
                f"{request_id}"
            )
        records.append(
            {
                "input_position": input_position,
                "request_id": request_id,
                "source_video_uid": str(source["source_video_uid"]),
                "video_path": str(source["video_path"]),
                "video_sha256": str(source["video_sha256"]),
                "source_duration_s": float(source["source_duration_s"]),
                "query": str(source["query"]),
                "gt_intervals": source.get("gt_intervals") or [],
                "direct_prompt": str(
                    source.get("direct_prompt_text")
                    or direct_prompt(str(source["query"]))
                ),
                "direct_target": direct_target,
                "direct_loss_prefix_text": str(
                    source.get("direct_loss_prefix_text") or ""
                ),
                "direct_lm_masked_char_spans": direct_lm_masked_char_spans,
                "direct_lm_weighted_char_spans": direct_lm_weighted_char_spans,
                "direct_duplicate_unlikelihood_char_spans": (
                    direct_duplicate_unlikelihood_char_spans
                ),
                "direct_endpoint_targets": [],
                "rollout_prompt": str(source["prompt_text"]),
                "rollout_response": response,
                "rollout_completion_ids": completion_ids,
                "rollout_strict_parse_valid": bool(parsed.parse_valid),
                "rollout_canonical_valid": bool(canonical.parse_valid),
                "rollout_confidence_available": confidence_available,
                "rollout_normalization_status": canonical.normalization_status,
                "rollout_targets": targets,
                "processor_contract": rollout["processor_contract"],
            }
        )

    source_videos = [record["source_video_uid"] for record in records]
    expected_unique_source_videos = int(
        manifest.get("unique_source_videos", expected_queries)
    )
    inline_confidence_queries = sum(
        bool(record["direct_endpoint_targets"]) for record in records
    )
    detached_confidence_replay_queries = sum(
        bool(record.get("confidence_replay_target")) for record in records
    )
    confidence_training_queries = (
        inline_confidence_queries + detached_confidence_replay_queries
        if confidence_kind in {"inline_verifier_pyes", "inline_gt_binary"}
        else expected_rollout_queries
    )
    confidence_disabled = confidence_kind == "none"
    checks = {
        "query_count_exact": len(records) == expected_queries,
        "request_units_unique": len({record["request_id"] for record in records})
        == expected_queries,
        "unique_source_video_count_exact": len(set(source_videos))
        == expected_unique_source_videos,
        "rollout_query_count_exact": len(output_by_id) == expected_rollout_queries,
        "inline_confidence_query_contract_exact": (
            (
                inline_confidence_queries == expected_queries
                and detached_confidence_replay_queries == 0
            )
            or (
                inline_confidence_queries == 0
                and detached_confidence_replay_queries == expected_queries
            )
            if confidence_kind in {"inline_verifier_pyes", "inline_gt_binary"}
            else inline_confidence_queries == detached_confidence_replay_queries == 0
        ),
        "confidence_endpoint_training_units_at_least_5000": (
            positive + negative >= MIN_CONFIDENCE_ENDPOINTS
            if not confidence_disabled
            else positive + negative == 0
        ),
        "positive_endpoints_at_least_1000": (
            positive >= 1_000 if not confidence_disabled else positive == 0
        ),
        "negative_endpoints_at_least_1000": (
            negative >= 1_000 if not confidence_disabled else negative == 0
        ),
        "all_queries_keep_gt_generation": all(record["direct_target"] for record in records),
        "rollout_lm_loss_masked_by_construction": (
            confidence_kind != "none" or expected_rollout_queries == 0
        ),
        "ordinary_gt_only_all_target_tokens": (
            confidence_kind == "none"
            and (
                manifest.get("protocol_version")
                in {
                    POSITIVE_ONLY_EXACT_DUPLICATE_PROTOCOL,
                    POSITIVE_ONLY_FIXED64_PROTOCOL,
                    POSITIVE_ONLY_FIXED16_HUNGARIAN_CONFIDENCE_PROTOCOL,
                    POSITIVE_ONLY_FIXED32_ALL_GT_RANDOM_NEGATIVE_CONFIDENCE_PROTOCOL,
                    POSITIVE_ONLY_FIXED32_GT_NMS_ANCHOR_COORDINATE_PROTOCOL,
                }
                or all(
                    not record["direct_lm_masked_char_spans"]
                    for record in records
                )
            )
        )
        if confidence_disabled
        else True,
    }
    if not all(checks.values()):
        raise SystemExit(f"joint training data gate failed: {checks}")
    summary = {
        "checks": checks,
        "queries": len(records),
        "confidence_training_queries": confidence_training_queries,
        "inline_confidence_queries": inline_confidence_queries,
        "detached_confidence_replay_queries": detached_confidence_replay_queries,
        "direct_only_queries": expected_queries - confidence_training_queries,
        "canonical_valid_confidence_queries": canonical_valid,
        "canonical_invalid_confidence_queries": expected_rollout_queries - canonical_valid,
        "duration_filtered_confidence_queries": duration_filtered,
        "strict_valid_before_duration_filter": sum(
            record["rollout_strict_parse_valid"] for record in records
        ),
        "endpoint_count": positive + negative,
        "positive_tiou05": positive,
        "negative_tiou05": negative,
        "positive_target05": positive,
        "negative_target05": negative,
        "confidence_target_kind": confidence_kind,
        "teacher_query_count": len(teacher_by_request),
        "teacher_candidate_count": (
            len(teacher_by_candidate)
            if confidence_kind == "inline_verifier_pyes"
            else sum(map(len, teacher_by_request.values()))
        ),
    }
    return records, summary


def video_messages(record: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": record["video_path"],
                    **VIDEO_CONTRACT,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def processor_contract(video_metadata: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    def get(name: str) -> Any:
        if isinstance(video_metadata, dict):
            return video_metadata[name]
        return getattr(video_metadata, name)

    grid = [int(value) for value in inputs["video_grid_thw"][0].tolist()]
    indices = [int(value) for value in get("frames_indices")]
    source_fps = float(get("fps"))
    merge_size = 2
    spatial_tokens = grid[1] * grid[2] // (merge_size * merge_size)
    return {
        "sampled_frames": len(indices),
        "sampled_source_frame_first": indices[0],
        "sampled_source_frame_last": indices[-1],
        "sampled_source_time_first_s": indices[0] / source_fps,
        "sampled_source_time_last_s": indices[-1] / source_fps,
        "source_fps": source_fps,
        "video_grid_thw": [grid],
        "visual_tokens_per_temporal_patch": spatial_tokens,
    }


def append_text_completion_inputs(
    prompt_inputs: dict[str, Any], completion_ids: Sequence[int]
) -> tuple[Any, Any, Any | None]:
    """Append assistant text while preserving Qwen3-VL modality type IDs."""

    import torch

    prompt_ids = prompt_inputs["input_ids"]
    completion = torch.tensor([list(completion_ids)], dtype=prompt_ids.dtype)
    full_ids = torch.cat([prompt_ids, completion], dim=1)
    prompt_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_ids))
    full_mask = torch.cat([prompt_mask, torch.ones_like(completion)], dim=1)
    prompt_types = prompt_inputs.get("mm_token_type_ids")
    full_types = None
    if prompt_types is not None:
        # 0 is ordinary text for Qwen3-VL; visual tokens retain their prompt IDs.
        full_types = torch.cat([prompt_types, torch.zeros_like(completion)], dim=1)
    return full_ids, full_mask, full_types


class QueryDataset:
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        mode: str,
        *,
        world_size: int,
        epochs: int,
        target_global_batch_queries: int,
    ) -> None:
        target_steps_per_epoch = math.ceil(
            len(records) / target_global_batch_queries
        )
        if mode == "direct":
            real = [(record, "direct") for record in records]
        elif mode == "joint_inline":
            if not all(record["direct_endpoint_targets"] for record in records):
                raise ValueError("joint-inline mode requires confidence on every direct row")
            real = [(record, "direct") for record in records]
        elif mode == "joint_interleaved":
            real = []
            for record in records:
                real.append((record, "direct"))
                if record["rollout_confidence_available"]:
                    real.append((record, "confidence"))
        else:
            raise ValueError(f"unsupported dataset mode: {mode}")

        # Make both arms exactly target_steps optimizer updates without letting
        # DistributedSampler duplicate real examples.  Zero-weight direct
        # padding is cheaper than a second confidence graph and keeps every DDP
        # parameter in the graph.  Scaling real losses by padded_total / 6500
        # makes the mean over the padded dataset exactly equal to the desired
        # per-query objective.
        self.real_micro_samples = len(real)
        self.target_optimizer_steps_per_epoch = target_steps_per_epoch
        self.target_optimizer_steps = target_steps_per_epoch * epochs
        self.gradient_accumulation_steps = math.ceil(
            self.real_micro_samples / (world_size * target_steps_per_epoch)
        )
        padded_total = (
            world_size
            * target_steps_per_epoch
            * self.gradient_accumulation_steps
        )
        self.padding_micro_samples = padded_total - self.real_micro_samples
        self.objective_branch_scale = padded_total / len(records)
        self.records = [
            {
                **record,
                "micro_branch": branch,
                "branch_scale": self.objective_branch_scale,
                "padding_micro_sample": False,
            }
            for record, branch in real
        ]
        for index in range(self.padding_micro_samples):
            self.records.append(
                {
                    **records[index % len(records)],
                    "micro_branch": "direct",
                    "branch_scale": 0.0,
                    "padding_micro_sample": True,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class PairedVideoCollator:
    def __init__(
        self, processor: Any, mode: str, endpoint_anchor: str = "after_span"
    ) -> None:
        if endpoint_anchor not in {"after_span", "after_end"}:
            raise ValueError(f"unsupported endpoint anchor: {endpoint_anchor}")
        self.processor = processor
        self.mode = mode
        self.endpoint_anchor = endpoint_anchor
        self.diagnostics_enabled = False
        self.local_video_root: Path | None = None
        if os.environ.get("TVG_LOCALIZE_VIDEO", "0") == "1":
            self.local_video_root = Path(
                os.environ.get("TVG_LOCAL_VIDEO_ROOT", "/tmp/tvg_video_cache")
            )
            self.local_video_root.mkdir(parents=True, exist_ok=True)
            from tvg_confidence.bounded_video_cache import BoundedVideoCache
            self.video_cache = BoundedVideoCache(
                self.local_video_root,
                max_bytes=int(os.environ.get("TVG_VIDEO_CACHE_MAX_BYTES", str(8 * 1024**3))),
                min_free_bytes=int(os.environ.get("TVG_VIDEO_CACHE_MIN_FREE_BYTES", str(8 * 1024**3))),
                report_path=os.environ.get("TVG_VIDEO_CACHE_REPORT"),
            )

    @contextmanager
    def _localize_video(self, source_value: str):
        """Lease a bounded local copy until video decoding has completed."""
        if self.local_video_root is None:
            yield source_value
            return
        source = Path(source_value)
        with self.video_cache.localize(source) as localized:
            yield localized

    def _decode_video(
        self, messages: list[dict[str, Any]]
    ) -> tuple[Any, Any, dict[str, Any]]:
        from qwen_vl_utils import process_vision_info

        started = time.monotonic()
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=int(getattr(self.processor.image_processor, "patch_size", 16)),
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is None or len(videos) != 1:
            raise ValueError("expected exactly one decoded video")
        videos, metadata = zip(*videos)
        if self.diagnostics_enabled:
            self._timings["decode_s"] = time.monotonic() - started
            self._timings["video_frames"] = int(videos[0].shape[0])
        return images, list(videos), {"metadata": list(metadata), **(video_kwargs or {})}

    def _process_text(
        self,
        rendered: str,
        images: Any,
        videos: Any,
        video_info: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs = dict(video_info)
        metadata = kwargs.pop("metadata")
        return dict(
            self.processor(
                text=[rendered],
                images=images,
                videos=videos,
                video_metadata=metadata,
                do_resize=False,
                return_tensors="pt",
                **kwargs,
            )
        )

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.monotonic()
        self._timings = {}
        result = self._collate(instances)
        if self.diagnostics_enabled:
            record = instances[0]
            direct = record.get("micro_branch", "direct") == "direct"
            cache = getattr(self, "video_cache", None)
            cache_metrics = dict(cache.last_metrics) if cache else {}
            elapsed = time.monotonic() - started
            result["_training_diagnostics"] = {
                **cache_metrics, **self._timings,
                "collate_s": elapsed,
                "preprocess_s": max(0, elapsed - self._timings.get("decode_s", 0) - self._timings.get("localize_s", 0)),
                "micro_samples": 0 if record.get("padding_micro_sample") else 1,
                "direct_queries": int(direct), "confidence_queries": int(not direct),
                "input_tokens": int(result["input_ids"].numel()),
                "gt_events": len(record.get("gt_intervals", [])) if direct else 0,
                "target_slots": len(json.loads(record["direct_target"])) if direct else 0,
            }
        return result

    def _collate(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if len(instances) != 1:
            raise ValueError("v1 freezes per-device micro-batch to one query")
        record = instances[0]
        branch = str(record.get("micro_branch", "direct"))
        if self.mode == "direct" and branch != "direct":
            raise ValueError("direct arm received a non-direct micro-branch")
        if branch not in {"direct", "confidence"}:
            raise ValueError(f"unsupported micro-branch: {branch}")
        prompt = record["direct_prompt"] if branch == "direct" else record["rollout_prompt"]
        localized_record = dict(record)
        localize_started = time.monotonic()
        with self._localize_video(str(record["video_path"])) as localized_path:
            if self.diagnostics_enabled:
                self._timings["localize_s"] = time.monotonic() - localize_started
            localized_record["video_path"] = localized_path
            prompt_user = video_messages(localized_record, prompt)
            images, videos, video_info = self._decode_video(prompt_user)

        prompt_rendered = self.processor.apply_chat_template(
            prompt_user, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self._process_text(
            prompt_rendered, images, videos, video_info
        )

        batch: dict[str, Any] = {
            "micro_branch": torch.tensor([0 if branch == "direct" else 1]),
            "branch_scale": torch.tensor(
                [float(record.get("branch_scale", 1.0))], dtype=torch.float32
            ),
        }
        for key, value in prompt_inputs.items():
            if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}:
                batch[f"visual__{key}"] = value

        if branch == "direct":
            # Append the assistant target to the exact generation prompt.
            # Each data protocol supplies its coordinate masks and weights;
            # the bounded-pool protocol supervises every target coordinate.
            # Separators and the final close remain supervised in all cases.
            direct_target = str(record["direct_target"])
            suffix_text = "<|im_end|>\n"
            masked_char_spans = list(
                record.get("direct_lm_masked_char_spans") or []
            )
            weighted_char_spans = list(
                record.get("direct_lm_weighted_char_spans") or []
            )
            duplicate_unlikelihood_char_spans = list(
                record.get("direct_duplicate_unlikelihood_char_spans") or []
            )
            direct_endpoint_targets = list(
                record.get("direct_endpoint_targets") or []
            )
            if (
                masked_char_spans
                or weighted_char_spans
                or duplicate_unlikelihood_char_spans
                or direct_endpoint_targets
            ):
                try:
                    target_encoding = self.processor.tokenizer(
                        direct_target,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                except (NotImplementedError, TypeError) as error:
                    raise ValueError(
                        "positive-only coverage SFT requires a fast tokenizer "
                        "with exact character offsets"
                    ) from error
                target_text_ids = list(target_encoding["input_ids"])
                target_offsets = [
                    (int(start), int(end))
                    for start, end in target_encoding["offset_mapping"]
                ]
                if len(target_text_ids) != len(target_offsets):
                    raise ValueError("target token/offset length mismatch")
            else:
                target_text_ids = list(
                    self.processor.tokenizer(
                        direct_target, add_special_tokens=False
                    )["input_ids"]
                )
                target_offsets = []
            suffix_ids = list(
                self.processor.tokenizer(
                    suffix_text, add_special_tokens=False
                )["input_ids"]
            )
            target_ids = [*target_text_ids, *suffix_ids]
            combined_target_ids = list(
                self.processor.tokenizer(
                    direct_target + suffix_text,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if target_ids != combined_target_ids:
                raise ValueError("assistant target tokenization changes at suffix boundary")
            if not target_ids:
                raise ValueError("direct GT assistant target tokenization is empty")
            completion_budget = int(record.get("completion_token_budget", 0))
            expected_completion = int(record.get("expected_completion_tokens", 0))
            if completion_budget and len(target_ids) > completion_budget:
                raise ValueError("candidate-pool completion exceeds its token budget")
            if expected_completion and len(target_ids) != expected_completion:
                raise ValueError("candidate-pool completion tokenizer length drift")
            expected_completion_sha = record.get("expected_completion_sha256")
            if expected_completion_sha and hashlib.sha256(
                json.dumps(target_ids, separators=(",", ":")).encode()
            ).hexdigest() != expected_completion_sha:
                raise ValueError("candidate-pool completion tokenizer identity drift")
            full_ids, full_mask, full_types = append_text_completion_inputs(
                prompt_inputs, target_ids
            )
            context_budget = int(record.get("context_token_budget", 0))
            if context_budget and int(full_ids.shape[1]) > context_budget:
                raise ValueError("video, prompt, and completion exceed the context token budget")
            prompt_length = int(prompt_inputs["input_ids"].shape[1])
            labels = torch.full_like(full_ids, -100)
            loss_prefix_text = str(record.get("direct_loss_prefix_text") or "")
            if loss_prefix_text and masked_char_spans:
                raise ValueError(
                    "prefix masking and arbitrary character-span masking are mutually exclusive"
                )
            if loss_prefix_text:
                if not direct_target.startswith(loss_prefix_text):
                    raise ValueError("direct LM-loss prefix is not a target prefix")
                prefix_ids = list(
                    self.processor.tokenizer(
                        loss_prefix_text,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                common = 0
                for left, right in zip(prefix_ids, target_ids):
                    if int(left) != int(right):
                        break
                    common += 1
                if common == 0 or common >= len(target_ids):
                    raise ValueError("direct LM-loss prefix has no usable token boundary")
                labels[:, prompt_length + common :] = full_ids[:, prompt_length + common :]
            else:
                labels[:, prompt_length:] = full_ids[:, prompt_length:]
                if masked_char_spans:
                    hit_counts = [0] * len(masked_char_spans)
                    for token_index, (token_start, token_end) in enumerate(target_offsets):
                        if token_end <= token_start:
                            continue
                        for span_index, (span_start, span_end) in enumerate(
                            masked_char_spans
                        ):
                            if max(token_start, span_start) < min(token_end, span_end):
                                labels[:, prompt_length + token_index] = -100
                                hit_counts[span_index] += 1
                    if any(hit == 0 for hit in hit_counts):
                        raise ValueError(
                            "an LM masked character span does not overlap a target token"
                        )
            batch.update(
                {"input_ids": full_ids, "attention_mask": full_mask, "labels": labels}
            )
            if weighted_char_spans:
                label_loss_weights = torch.zeros_like(full_ids, dtype=torch.float32)
                label_loss_weights[labels != -100] = 1.0
                label_loss_groups = torch.full_like(full_ids, 2)  # Grammar and assistant suffix.
                hit_counts = [0] * len(weighted_char_spans)
                for token_index, (token_start, token_end) in enumerate(target_offsets):
                    if token_end <= token_start:
                        continue
                    for span_index, item in enumerate(weighted_char_spans):
                        span_start, span_end = map(int, item["char_span"])
                        if max(token_start, span_start) < min(token_end, span_end):
                            position = prompt_length + token_index
                            if int(labels[0, position]) == -100:
                                raise ValueError(
                                    "an LM weighted character span overlaps a masked token"
                                )
                            label_loss_weights[0, position] = float(item["weight"])
                            if self.diagnostics_enabled:
                                label_loss_groups[0, position] = {"gt": 0, "alternative": 1}[item["kind"]]
                            hit_counts[span_index] += 1
                if any(hit == 0 for hit in hit_counts):
                    raise ValueError(
                        "an LM weighted character span does not overlap a target token"
                    )
                batch["label_loss_weights"] = label_loss_weights
                if self.diagnostics_enabled:
                    batch["label_loss_groups"] = label_loss_groups
            if duplicate_unlikelihood_char_spans:
                duplicate_token_indices: list[int] = []
                hit_counts = [0] * len(duplicate_unlikelihood_char_spans)
                for token_index, (token_start, token_end) in enumerate(target_offsets):
                    if token_end <= token_start:
                        continue
                    for span_index, (span_start, span_end) in enumerate(
                        duplicate_unlikelihood_char_spans
                    ):
                        if max(token_start, span_start) < min(token_end, span_end):
                            duplicate_token_indices.append(token_index)
                            hit_counts[span_index] += 1
                            break
                if any(hit == 0 for hit in hit_counts):
                    raise ValueError(
                        "an exact-duplicate unlikelihood span does not overlap a target token"
                    )
                absolute_positions = [
                    prompt_length + token_index
                    for token_index in duplicate_token_indices
                ]
                if any(position <= 0 for position in absolute_positions):
                    raise ValueError("invalid exact-duplicate prediction position")
                batch.update(
                    {
                        "duplicate_unlikelihood_positions": torch.tensor(
                            [absolute_positions], dtype=torch.long
                        ),
                        "duplicate_unlikelihood_token_ids": full_ids.index_select(
                            1,
                            torch.tensor(absolute_positions, dtype=torch.long),
                        ),
                    }
                )
            if direct_endpoint_targets:
                endpoint_positions: list[int] = []
                for target in direct_endpoint_targets:
                    span_start, span_end = map(int, target["end_char_span"])
                    overlapping = [
                        token_index
                        for token_index, (token_start, token_end) in enumerate(
                            target_offsets
                        )
                        if max(token_start, span_start) < min(token_end, span_end)
                    ]
                    if not overlapping:
                        raise ValueError(
                            "a direct confidence endpoint does not overlap a target token"
                        )
                    endpoint_positions.append(prompt_length + max(overlapping))
                batch.update(
                    {
                        "endpoint_positions": torch.tensor(
                            [endpoint_positions], dtype=torch.long
                        ),
                        "endpoint_quality": torch.tensor(
                            [[
                                target["max_direct_tiou"]
                                for target in direct_endpoint_targets
                            ]],
                            dtype=torch.float32,
                        ),
                        "endpoint_ordinal": torch.tensor(
                            [[
                                target["ordinal"]
                                for target in direct_endpoint_targets
                            ]],
                            dtype=torch.float32,
                        ),
                    }
                )
            if full_types is not None:
                batch["mm_token_type_ids"] = full_types
        else:
            if self.mode != "joint" or not record["rollout_confidence_available"]:
                raise ValueError("confidence micro-branch lacks a canonical-valid rollout")
            confidence_replay_target = str(
                record.get("confidence_replay_target") or ""
            )
            if confidence_replay_target:
                replay_targets = list(record.get("rollout_targets") or [])
                if not replay_targets:
                    raise ValueError("detached confidence replay has no candidate targets")
                try:
                    replay_encoding = self.processor.tokenizer(
                        confidence_replay_target,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                except (NotImplementedError, TypeError) as error:
                    raise ValueError(
                        "detached span-confidence replay requires exact tokenizer offsets"
                    ) from error
                replay_text_ids = list(replay_encoding["input_ids"])
                replay_offsets = [
                    (int(start), int(end))
                    for start, end in replay_encoding["offset_mapping"]
                ]
                suffix_text = "<|im_end|>\n"
                suffix_ids = list(
                    self.processor.tokenizer(
                        suffix_text, add_special_tokens=False
                    )["input_ids"]
                )
                replay_ids = [*replay_text_ids, *suffix_ids]
                replay_budget = int(record.get("replay_completion_token_budget", 0))
                if replay_budget and len(replay_ids) > replay_budget:
                    raise ValueError("confidence replay exceeds its completion budget")
                expected_replay = int(record.get("expected_replay_tokens", 0))
                if expected_replay and len(replay_ids) != expected_replay:
                    raise ValueError("confidence replay tokenizer length drift")
                expected_replay_sha = record.get("expected_replay_sha256")
                if expected_replay_sha and hashlib.sha256(
                    json.dumps(replay_ids, separators=(",", ":")).encode()
                ).hexdigest() != expected_replay_sha:
                    raise ValueError("confidence replay tokenizer identity drift")
                if replay_ids != list(
                    self.processor.tokenizer(
                        confidence_replay_target + suffix_text,
                        add_special_tokens=False,
                    )["input_ids"]
                ):
                    raise ValueError(
                        "confidence replay tokenization changes at suffix boundary"
                    )
                full_ids, full_mask, full_types = append_text_completion_inputs(
                    prompt_inputs, replay_ids
                )
                context_budget = int(record.get("context_token_budget", 0))
                if context_budget and int(full_ids.shape[1]) > context_budget:
                    raise ValueError("video, prompt, and confidence replay exceed the context budget")
                prompt_length = int(prompt_inputs["input_ids"].shape[1])
                span_token_positions: list[int] = []
                span_token_candidate_indices: list[int] = []
                for candidate_index, target in enumerate(replay_targets):
                    span_start, span_end = map(int, target["span_char_span"])
                    positions = [
                        prompt_length + token_index
                        for token_index, (token_start, token_end) in enumerate(
                            replay_offsets
                        )
                        if token_end > token_start
                        and max(token_start, span_start) < min(token_end, span_end)
                    ]
                    if not positions:
                        raise ValueError(
                            "a detached confidence span does not overlap a target token"
                        )
                    span_token_positions.extend(positions)
                    span_token_candidate_indices.extend(
                        [candidate_index] * len(positions)
                    )
                batch.update(
                    {
                        "input_ids": full_ids,
                        "attention_mask": full_mask,
                        "labels": torch.full_like(full_ids, -100),
                        "span_token_positions": torch.tensor(
                            [span_token_positions], dtype=torch.long
                        ),
                        "span_token_candidate_indices": torch.tensor(
                            [span_token_candidate_indices], dtype=torch.long
                        ),
                        "span_candidate_count": torch.tensor(
                            [len(replay_targets)], dtype=torch.long
                        ),
                        "endpoint_quality": torch.tensor(
                            [[
                                target["max_direct_tiou"]
                                for target in replay_targets
                            ]],
                            dtype=torch.float32,
                        ),
                    }
                )
                if full_types is not None:
                    batch["mm_token_type_ids"] = full_types
                return batch
            completion_ids = record["rollout_completion_ids"]
            decoded = self.processor.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if decoded != record["rollout_response"]:
                raise ValueError("rollout completion IDs do not decode exactly")
            canonical = canonicalize_interval_set_to_duration(
                decoded, duration_s=float(record["source_duration_s"])
            )
            parsed = canonical.source_parse
            token_rows, anchor_rows, mapping_status = map_token_semantics(
                self.processor.tokenizer, completion_ids, decoded, parsed
            )
            del token_rows
            endpoint_anchors = {
                int(anchor["span_index"]): int(anchor["prefix_token_end"])
                for anchor in anchor_rows
                if anchor.get("semantic_event") == self.endpoint_anchor
                and anchor.get("span_index") is not None
                and isinstance(anchor.get("prefix_token_end"), int)
            }
            if (
                mapping_status != "exact"
                or len(endpoint_anchors) != len(parsed.intervals)
                or sorted(endpoint_anchors) != list(range(len(endpoint_anchors)))
            ):
                raise ValueError(
                    f"rollout {self.endpoint_anchor} mapping is not exact"
                )
            rollout_ids, rollout_mask, rollout_types = append_text_completion_inputs(
                prompt_inputs, completion_ids
            )
            prompt_length = int(prompt_inputs["input_ids"].shape[1])
            source_span_indices = [
                int(target["source_span_index"])
                for target in record["rollout_targets"]
            ]
            if source_span_indices != canonical.source_span_indices:
                raise ValueError("rollout duration-filter lineage drift")
            positions = [
                prompt_length + endpoint_anchors[index]
                for index in source_span_indices
            ]
            if any(position >= rollout_ids.shape[1] for position in positions):
                raise ValueError("rollout endpoint position is out of range")
            batch.update(
                {
                    "input_ids": rollout_ids,
                    "attention_mask": rollout_mask,
                    # Transformers 4.57.1 inspects ``labels`` across every
                    # gradient-accumulation micro-batch before calling our
                    # custom compute_loss.  Confidence-only samples have no LM
                    # targets, but still need an all-ignore tensor so that the
                    # Trainer can count zero supervised language tokens.
                    "labels": torch.full_like(rollout_ids, -100),
                    "endpoint_positions": torch.tensor([positions], dtype=torch.long),
                    "endpoint_quality": torch.tensor(
                        [[target["max_direct_tiou"] for target in record["rollout_targets"]]],
                        dtype=torch.float32,
                    ),
                    "endpoint_ordinal": torch.tensor(
                        [[target["ordinal"] for target in record["rollout_targets"]]],
                        dtype=torch.float32,
                    ),
                }
            )
            if rollout_types is not None:
                batch["mm_token_type_ids"] = rollout_types
        return batch


def base_generator(model: Any) -> Any:
    generator = model.generator
    return generator.get_base_model() if hasattr(generator, "get_base_model") else generator


class SingleStageConfidenceModel:
    """Factory wrapper so torch is imported only inside the GPU runtime."""

    @staticmethod
    def build(
        generator: Any,
        confidence_head: Any,
        mode: str,
        lambda_conf: float,
        lambda_exact_duplicate: float,
        confidence_backprop_to_generator: bool,
        generator_frozen: bool,
        confidence_ranking_weight: float = 0.25,
    ) -> Any:
        import torch
        from torch.utils.checkpoint import checkpoint

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.generator = generator
                self.confidence_head = confidence_head
                self.mode = mode
                self.lambda_conf = float(lambda_conf)
                self.lambda_exact_duplicate = float(lambda_exact_duplicate)
                self.confidence_backprop_to_generator = bool(
                    confidence_backprop_to_generator
                )
                self.generator_frozen = bool(generator_frozen)
                self.diagnostics_enabled = False

            def _backbone(self) -> Any:
                return base_generator(self)

            @staticmethod
            def _visual(kwargs: dict[str, Any]) -> dict[str, Any]:
                return {
                    key.removeprefix("visual__"): value
                    for key, value in kwargs.items()
                    if key.startswith("visual__")
                }

            def _base_forward(
                self,
                input_ids: Any,
                attention_mask: Any,
                mm_token_type_ids: Any,
                visual: dict[str, Any],
            ) -> Any:
                backbone = self._backbone()
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                    "return_dict": True,
                    **visual,
                }
                # Transformers 4.57.1 (the frozen H20 runtime) derives
                # multimodal RoPE positions from the placeholder token IDs and
                # does not emit this field.  Newer releases may emit it; only
                # forward the field when the processor actually supplied it.
                if mm_token_type_ids is not None:
                    model_inputs["mm_token_type_ids"] = mm_token_type_ids
                if self.generator_frozen:
                    # The strict head-only comparison preserves the direct
                    # generator byte-for-byte.  Avoid retaining an activation
                    # graph through the frozen 4B backbone; gradients begin at
                    # the independent confidence head below.
                    with torch.no_grad():
                        return backbone.model(
                            **model_inputs,
                        ).last_hidden_state
                return backbone.model(**model_inputs).last_hidden_state

            def _confidence_objective(
                self, hidden: Any, kwargs: dict[str, Any]
            ) -> tuple[Any, Any, Any, Any, Any, Any]:
                if "span_token_positions" in kwargs:
                    positions = kwargs["span_token_positions"][0]
                    candidate_indices = kwargs["span_token_candidate_indices"][0]
                    candidate_count = int(kwargs["span_candidate_count"].item())
                    token_hidden = hidden[0].index_select(0, positions)
                    span_hidden = token_hidden.new_zeros(
                        (candidate_count, token_hidden.shape[-1])
                    )
                    span_hidden.index_add_(0, candidate_indices, token_hidden)
                    counts = torch.bincount(
                        candidate_indices, minlength=candidate_count
                    ).to(token_hidden.dtype)
                    if bool((counts <= 0).any()):
                        raise RuntimeError("detached confidence replay has an empty span")
                    span_hidden = span_hidden / counts[:, None]
                    generator_graph_anchor = span_hidden.sum() * 0.0
                    if not self.confidence_backprop_to_generator:
                        span_hidden = span_hidden.detach()
                    logits = self.confidence_head(span_hidden.float()).float().squeeze(-1)
                    targets = kwargs["endpoint_quality"][0]
                    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets
                    )
                    differences = targets[:, None] - targets[None, :]
                    better, worse = torch.where(differences >= 0.1)
                    if self.diagnostics_enabled:
                        from tvg_confidence.training_diagnostics import confidence_statistics
                        self._diagnostics.update(confidence_statistics(logits, targets, better, worse))
                    pair_loss = generator_graph_anchor
                    if confidence_ranking_weight > 0 and len(better) > 0:
                        pair_loss = torch.nn.functional.softplus(
                            -(logits[better] - logits[worse])
                        ).mean()
                    confidence_loss = (
                        binary_loss + confidence_ranking_weight * pair_loss + generator_graph_anchor
                    )
                    zero = generator_graph_anchor
                    return (
                        confidence_loss,
                        binary_loss,
                        zero,
                        zero,
                        pair_loss,
                        generator_graph_anchor,
                    )
                positions = kwargs["endpoint_positions"][0]
                endpoint_hidden = hidden[0].index_select(0, positions)
                generator_graph_anchor = endpoint_hidden.sum() * 0.0
                if not self.confidence_backprop_to_generator:
                    endpoint_hidden = endpoint_hidden.detach()
                head_logits = self.confidence_head(endpoint_hidden.float()).float()
                quality = torch.sigmoid(head_logits[:, 0])
                ordinal_probability = torch.sigmoid(head_logits[:, 1:])
                targets = kwargs["endpoint_quality"][0]
                ordinal_targets = kwargs["endpoint_ordinal"][0]
                regression_loss = torch.nn.functional.smooth_l1_loss(
                    quality, targets, beta=0.1
                )
                ordinal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    head_logits[:, 1:], ordinal_targets
                )
                monotonic_loss = (
                    torch.relu(
                        ordinal_probability[:, 1] - ordinal_probability[:, 0]
                    ).mean()
                    + torch.relu(
                        ordinal_probability[:, 2] - ordinal_probability[:, 1]
                    ).mean()
                )
                differences = targets[:, None] - targets[None, :]
                better, worse = torch.where(differences >= 0.1)
                pair_loss = generator_graph_anchor
                if len(better) > 0:
                    pair_loss = torch.nn.functional.softplus(
                        -(head_logits[better, 0] - head_logits[worse, 0])
                    ).mean()
                confidence_loss = (
                    regression_loss
                    + 0.75 * ordinal_loss
                    + 0.1 * monotonic_loss
                    + 0.25 * pair_loss
                    + generator_graph_anchor
                )
                return (
                    confidence_loss,
                    regression_loss,
                    ordinal_loss,
                    monotonic_loss,
                    pair_loss,
                    generator_graph_anchor,
                )

            def forward(self, **kwargs: Any) -> dict[str, Any]:
                self._diagnostics = {}
                visual = self._visual(kwargs)
                branch = int(kwargs["micro_branch"].item())
                hidden = self._base_forward(
                    kwargs["input_ids"],
                    kwargs["attention_mask"],
                    kwargs.get("mm_token_type_ids"),
                    visual,
                )
                zero = hidden.sum() * 0.0
                lm_loss = zero
                regression_loss = zero
                ordinal_loss = zero
                monotonic_loss = zero
                pair_loss = zero
                confidence_loss = zero
                duplicate_unlikelihood_loss = zero
                if branch == 0:
                    if not self.generator_frozen:
                        shifted_labels = kwargs["labels"][:, 1:]
                        prediction_states = hidden[:, :-1, :]
                        mask = shifted_labels != -100
                        if int(mask.sum()) == 0:
                            raise RuntimeError(
                                "GT generation branch has no supervised token"
                            )
                        backbone = self._backbone()
                        supervised_states = prediction_states[mask]
                        supervised_labels = shifted_labels[mask]
                        shifted_loss_weights = kwargs.get("label_loss_weights")
                        supervised_loss_weights = None
                        if shifted_loss_weights is not None:
                            shifted_loss_weights = shifted_loss_weights[:, 1:]
                            supervised_loss_weights = shifted_loss_weights[mask].float()
                            if (
                                supervised_loss_weights.numel()
                                != supervised_labels.numel()
                                or not torch.isfinite(supervised_loss_weights).all()
                                or bool((supervised_loss_weights <= 0).any())
                            ):
                                raise RuntimeError("invalid supervised LM token weights")

                        def checkpointed_cross_entropy_sum(
                            states: Any, labels: Any, weights: Any, groups: Any
                        ) -> Any:
                            logits = backbone.lm_head(states).float()
                            losses = torch.nn.functional.cross_entropy(
                                logits, labels, reduction="none"
                            )
                            if self.diagnostics_enabled:
                                from tvg_confidence.training_diagnostics import coordinate_statistics
                                statistics = coordinate_statistics(logits, labels, losses, weights, groups)
                            else:
                                statistics = losses.new_zeros(12)
                            return (losses * weights).sum(), statistics

                        loss_sum = zero
                        coordinate_sums = hidden.new_zeros(12, dtype=torch.float32)
                        supervised_groups = kwargs.get("label_loss_groups", torch.full_like(kwargs["labels"], 2))[:, 1:][mask]
                        for start in range(
                            0,
                            int(supervised_labels.shape[0]),
                            LM_HEAD_CHECKPOINT_CHUNK_TOKENS,
                        ):
                            end = start + LM_HEAD_CHECKPOINT_CHUNK_TOKENS
                            chunk_weights = (
                                supervised_loss_weights[start:end]
                                if supervised_loss_weights is not None
                                else torch.ones_like(
                                    supervised_labels[start:end], dtype=torch.float32
                                )
                            )
                            chunk_loss, chunk_statistics = checkpoint(
                                checkpointed_cross_entropy_sum,
                                supervised_states[start:end],
                                supervised_labels[start:end],
                                chunk_weights,
                                supervised_groups[start:end],
                                use_reentrant=False,
                            )
                            loss_sum = loss_sum + chunk_loss
                            coordinate_sums = coordinate_sums + chunk_statistics
                        loss_denominator = (
                            supervised_loss_weights.sum()
                            if supervised_loss_weights is not None
                            else supervised_labels.new_tensor(
                                supervised_labels.numel(), dtype=torch.float32
                            )
                        )
                        lm_loss = loss_sum / loss_denominator.clamp(min=1.0)
                        if self.diagnostics_enabled:
                            for index, group in enumerate(("gt", "alternative", "format")):
                                values = coordinate_sums[index * 4:index * 4 + 4]
                                self._diagnostics.update({f"ce_{group}_sum": values[0],
                                    f"tokens_{group}": values[1], f"correct_{group}": values[2],
                                    f"weighted_ce_{group}_sum": values[3] / loss_denominator.detach().clamp(min=1.0)})
                        del prediction_states, supervised_states, supervised_labels
                        if supervised_loss_weights is not None:
                            del supervised_loss_weights
                        if "duplicate_unlikelihood_positions" in kwargs:
                            duplicate_positions = kwargs[
                                "duplicate_unlikelihood_positions"
                            ][0]
                            duplicate_token_ids = kwargs[
                                "duplicate_unlikelihood_token_ids"
                            ][0]
                            duplicate_states = hidden[0].index_select(
                                0, duplicate_positions - 1
                            )
                            duplicate_logits = backbone.lm_head(
                                duplicate_states
                            ).float()
                            duplicate_log_probability = (
                                torch.gather(
                                    torch.nn.functional.log_softmax(
                                        duplicate_logits, dim=-1
                                    ),
                                    1,
                                    duplicate_token_ids[:, None],
                                )
                                .squeeze(1)
                                .clamp(max=-1e-7)
                            )
                            duplicate_unlikelihood_loss = -torch.log1p(
                                -duplicate_log_probability.exp()
                            ).mean()
                            del duplicate_states, duplicate_logits
                    if self.mode == "joint" and "endpoint_positions" in kwargs:
                        (
                            confidence_loss,
                            regression_loss,
                            ordinal_loss,
                            monotonic_loss,
                            pair_loss,
                            _,
                        ) = self._confidence_objective(hidden, kwargs)
                    elif self.mode == "joint":
                        # Every rank sees the same interleaved branch schedule;
                        # keep the head in DDP's graph on direct micro-samples.
                        lm_loss = lm_loss + sum(
                            parameter.sum() * 0.0
                            for parameter in self.confidence_head.parameters()
                        )
                elif branch == 1 and self.mode == "joint":
                    (
                        confidence_loss,
                        regression_loss,
                        ordinal_loss,
                        monotonic_loss,
                        pair_loss,
                        _,
                    ) = self._confidence_objective(hidden, kwargs)
                else:
                    raise RuntimeError(f"invalid model branch/mode: {branch}/{self.mode}")

                del hidden
                # Interleaved legacy data uses separate direct/confidence
                # micro-samples.  Positive-only coverage data attaches its
                # endpoint targets to the direct teacher-forced sequence, so
                # the same graph computes LM + lambda_conf * confidence and
                # avoids a second video decode.
                branch_scale = kwargs.get("branch_scale")
                if branch_scale is None:
                    branch_scale = lm_loss.new_tensor(1.0)
                loss = branch_scale.reshape(()) * (
                    (
                        lm_loss
                        + self.lambda_exact_duplicate
                        * duplicate_unlikelihood_loss
                        + self.lambda_conf * confidence_loss
                        if branch == 0
                        else self.lambda_conf * confidence_loss
                    )
                )
                if self.diagnostics_enabled:
                    self._diagnostics.update({"lm_loss_sum": lm_loss.detach(),
                        "confidence_loss_sum": confidence_loss.detach(), "soft_bce_sum": regression_loss.detach(),
                        "ranking_loss_sum": pair_loss.detach(),
                        "objective_lm_sum": branch_scale.reshape(()) * lm_loss.detach(),
                        "objective_confidence_sum": branch_scale.reshape(()) * self.lambda_conf * confidence_loss.detach(),
                        "nonfinite_objectives": (~torch.isfinite(loss.detach())).float()})
                return {
                    "loss": loss,
                    "diagnostics": self._diagnostics,
                    "lm_loss": lm_loss.detach(),
                    "duplicate_unlikelihood_loss": (
                        duplicate_unlikelihood_loss.detach()
                    ),
                    "confidence_loss": confidence_loss.detach(),
                    "regression_loss": regression_loss.detach(),
                    "ordinal_loss": ordinal_loss.detach(),
                    "monotonic_loss": monotonic_loss.detach(),
                    "pair_loss": pair_loss.detach(),
                }

        return Model()


def make_generator(
    model_path: str,
    revision: str,
    initialization: dict[str, Any] | None = None,
) -> Any:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText

    stagger_seconds = float(os.environ.get("TVG_MODEL_LOAD_STAGGER_SECONDS", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if stagger_seconds > 0.0 and local_rank > 0:
        delay = local_rank * stagger_seconds
        print(
            json.dumps(
                {
                    "stage": "model_load_stagger",
                    "local_rank": local_rank,
                    "seconds": delay,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(delay)
    generator = AutoModelForImageTextToText.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    generator.config.use_cache = False
    generator.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    generator.enable_input_require_grads()
    if initialization is None:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules=list(LORA_TARGETS),
        )
        generator = get_peft_model(generator, lora)
    else:
        if initialization.get("kind") != "peft_lora_continuation":
            raise RuntimeError(
                f"unsupported generator initialization: {initialization.get('kind')}"
            )
        adapter_dir = Path(str(initialization["adapter_dir"]))
        adapter_model = adapter_dir / "adapter_model.safetensors"
        adapter_config = adapter_dir / "adapter_config.json"
        expected = {
            adapter_model: str(initialization["adapter_model_sha256"]),
            adapter_config: str(initialization["adapter_config_sha256"]),
        }
        for path, expected_sha256 in expected.items():
            if not path.is_file() or sha256_file(path) != expected_sha256:
                raise RuntimeError(f"generator initialization mismatch: {path}")
        generator = PeftModel.from_pretrained(
            generator,
            str(adapter_dir),
            is_trainable=True,
        )
    for name, parameter in generator.named_parameters():
        if "visual" in name or "vision" in name:
            parameter.requires_grad_(False)
    trainable = [name for name, parameter in generator.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora" not in name.lower() for name in trainable):
        raise RuntimeError("unexpected trainable generator parameter set")
    if any("visual" in name or "vision" in name for name in trainable):
        raise RuntimeError("vision parameters must remain frozen")
    return generator


def localize_model_directory(
    source_path: str,
    revision: str,
    rank: int,
) -> Path:
    """Copy an OSS-mounted checkpoint once before all ranks mmap its shards.

    GPU exposes the model through an shared filesystem mount.  Concurrent safetensors
    mmap reads from all eight ranks can terminate one rank with SIGBUS.  The
    worker-local filesystem is shared by the eight processes, so rank zero
    performs one checked copy and the remaining ranks wait at a process-group
    barrier before loading from local disk.
    """

    source = Path(source_path)
    if os.environ.get("TVG_LOCALIZE_MODEL", "0") != "1":
        return source

    import torch.distributed as distributed

    cache_root = Path(
        os.environ.get("TVG_LOCAL_MODEL_ROOT", "/tmp/tvg_model_cache")
    )
    target = cache_root / f"{source.name}-{revision[:12]}"
    marker = target / ".tvg_model_ready.json"
    if rank == 0 and not marker.is_file():
        expected_sizes = {
            str(path.relative_to(source)): path.stat().st_size
            for path in source.rglob("*")
            if path.is_file()
        }
        if not expected_sizes or not any(
            name.endswith(".safetensors") for name in expected_sizes
        ):
            raise RuntimeError(f"model source is incomplete: {source}")
        cache_root.mkdir(parents=True, exist_ok=True)
        staging = cache_root / f".{target.name}.staging-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        observed_sizes = {
            str(path.relative_to(staging)): path.stat().st_size
            for path in staging.rglob("*")
            if path.is_file()
        }
        if observed_sizes != expected_sizes:
            raise RuntimeError("localized model file-size manifest mismatch")
        write_json(
            staging / marker.name,
            {
                "source_path": str(source),
                "revision": revision,
                "files": len(expected_sizes),
                "bytes": sum(expected_sizes.values()),
            },
        )
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        print(
            json.dumps(
                {
                    "stage": "model_localization_complete",
                    "source": str(source),
                    "target": str(target),
                    "files": len(expected_sizes),
                    "bytes": sum(expected_sizes.values()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if distributed.is_available() and distributed.is_initialized():
        distributed.barrier()
    if not marker.is_file():
        raise RuntimeError(f"localized model completion marker missing: {marker}")
    receipt = json.loads(marker.read_text())
    if (
        receipt.get("source_path") != str(source)
        or receipt.get("revision") != revision
    ):
        raise RuntimeError("localized model completion marker mismatch")
    return target


def _new_confidence_head() -> Any:
    import torch

    return torch.nn.Sequential(
        torch.nn.LayerNorm(2_560),
        torch.nn.Linear(2_560, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 4),
    )


def _new_span_confidence_head() -> Any:
    import torch

    return torch.nn.Sequential(
        torch.nn.LayerNorm(2_560),
        torch.nn.Linear(2_560, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 1),
    )


def make_confidence_head(spec: dict[str, Any]) -> Any:
    import torch

    kind = str(spec.get("kind"))
    if kind == "random_span_hidden_binary_mlp":
        architecture = spec.get("architecture", {})
        if architecture != {
            "input_dim": 2_560,
            "hidden_dim": 256,
            "output_dim": 1,
            "dropout": 0.1,
        }:
            raise SystemExit("random span-confidence architecture mismatch")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec["seed"]))
            return _new_span_confidence_head()
    if kind == "random_mlp":
        architecture = spec.get("architecture", {})
        if architecture != {
            "input_dim": 2_560,
            "hidden_dim": 256,
            "output_dim": 4,
            "dropout": 0.1,
        }:
            raise SystemExit("random confidence-head architecture mismatch")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec["seed"]))
            return _new_confidence_head()
    if kind != "independently_confirmed_hidden_mlp_run03":
        raise SystemExit(f"unsupported confidence-head initialization: {kind}")
    from safetensors.torch import load_file

    checkpoint = Path(spec["path"])
    if sha256_file(checkpoint) != str(spec["sha256"]):
        raise SystemExit("confidence-head checkpoint SHA mismatch")
    head = _new_confidence_head()
    head.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    return head


def train_arm(
    *,
    arm: str,
    records: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    processor: Any,
    output_root: Path,
    lambda_conf: float,
) -> dict[str, Any]:
    import torch
    from transformers import Trainer, TrainingArguments

    random.seed(20260813)
    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)
    generator = make_generator(
        str(manifest["model"]["path"]),
        str(manifest["model"]["revision"]),
        manifest.get("generator_initialization"),
    )
    generator_update_mode = str(
        manifest.get("training_contract", {}).get(
            "generator_update_mode", "lm_only_with_detached_confidence"
        )
    )
    if generator_update_mode not in {
        "lm_only_with_detached_confidence",
        "lm_only_with_all_coordinates_supervised",
        "frozen_head_only",
    }:
        raise RuntimeError(
            f"unsupported generator_update_mode: {generator_update_mode}"
        )
    generator_frozen = generator_update_mode == "frozen_head_only"
    if generator_frozen:
        if arm != "joint":
            raise RuntimeError("frozen_head_only is only valid for the joint arm")
        for parameter in generator.parameters():
            parameter.requires_grad_(False)
        if hasattr(generator, "disable_input_require_grads"):
            generator.disable_input_require_grads()
        generator.gradient_checkpointing_disable()
        if any(parameter.requires_grad for parameter in generator.parameters()):
            raise RuntimeError("frozen_head_only left trainable generator parameters")
    confidence_head = None
    if arm == "joint":
        head_spec = manifest["head_initialization"]
        confidence_head = make_confidence_head(head_spec)
    else:
        confidence_head = torch.nn.Identity()
    confidence_backprop_to_generator = bool(
        manifest.get("training_contract", {}).get(
            "confidence_backprop_to_generator", True
        )
    )
    lambda_exact_duplicate = float(
        manifest.get("training_contract", {}).get(
            "lambda_exact_duplicate_unlikelihood", 0.0
        )
    )
    if lambda_exact_duplicate < 0.0:
        raise RuntimeError("exact-duplicate unlikelihood weight must be non-negative")
    model = SingleStageConfidenceModel.build(
        generator,
        confidence_head,
        arm,
        lambda_conf if arm == "joint" else 0.0,
        lambda_exact_duplicate if arm == "direct" else 0.0,
        confidence_backprop_to_generator,
        generator_frozen,
        confidence_ranking_weight=float(manifest["training_contract"].get("confidence_ranking_weight", 0.25)),
    )
    inline_confidence = (
        manifest.get("training_contract", {}).get("confidence_supervision_mode")
        == "inline_direct_target"
    )
    dataset_mode = (
        "direct"
        if arm == "direct"
        else "joint_inline"
        if inline_confidence
        else "joint_interleaved"
    )
    world_size = int(
        os.environ.get(
            "TVG_EFFECTIVE_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")
        )
    )
    contract = manifest["training_contract"]
    epochs = int(contract["epochs_per_arm"])
    target_global_batch_queries = int(contract["target_global_batch_queries"])
    train_dataset = QueryDataset(
        records,
        dataset_mode,
        world_size=world_size,
        epochs=epochs,
        target_global_batch_queries=target_global_batch_queries,
    )
    micro_sample_ratio = len(train_dataset) / len(records)
    gradient_accumulation_steps = train_dataset.gradient_accumulation_steps

    from tvg_confidence.compact_training_checkpoints import (
        CompactCheckpointMixin, checkpoint_contract, latest_checkpoint, progress_callback,
    )
    full_soft = manifest["protocol_version"] in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS

    from tvg_confidence.training_diagnostics import TrainingDiagnosticsMixin

    class ConfidenceTrainer(TrainingDiagnosticsMixin, CompactCheckpointMixin, Trainer):
        def create_optimizer(self) -> Any:
            if self.optimizer is not None:
                return self.optimizer
            adapter_parameters = [
                parameter
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and "confidence_head" not in name
            ]
            groups: list[dict[str, Any]] = []
            if adapter_parameters:
                groups.append(
                    {
                        "params": adapter_parameters,
                        "lr": 2e-5,
                        "weight_decay": 0.01,
                    }
                )
            if arm == "joint":
                head_parameters = [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad and "confidence_head" in name
                ]
                if not head_parameters:
                    raise RuntimeError("joint arm has no trainable confidence head")
                groups.append(
                    {
                        "params": head_parameters,
                        "lr": 2e-4,
                        "weight_decay": 0.01,
                    }
                )
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
            return self.optimizer

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            outputs = model(**inputs)
            self.add_diagnostics(inputs, outputs)
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    arm_dir = output_root / arm
    arguments = TrainingArguments(
        output_dir=str(arm_dir / "trainer-state"),
        overwrite_output_dir=False,
        num_train_epochs=float(epochs),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps" if full_soft else "no",
        save_steps=int(contract.get("checkpoint_steps", 200)),
        save_total_limit=int(contract.get("checkpoint_total_limit", 3)),
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        # Video decoding already runs once per DDP rank.  Forking two loader
        # workers per rank creates 16 concurrent decoders and, in the H20
        # runtime, one worker was killed after step 8 before NCCL aborted the
        # remaining ranks.  Keep decoding in the rank process so the retry
        # changes execution only, not the frozen samples or objective.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        seed=20260813,
        data_seed=20260813,
    )
    trainer = ConfidenceTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        data_collator=PairedVideoCollator(
            processor,
            arm,
            str(manifest["training_contract"].get("endpoint_anchor", "after_span")),
        ),
    )
    resume_checkpoint = None
    diagnostic_config = contract.get("training_diagnostics", {})
    trainer.diagnostic_every = int(diagnostic_config.get("every_steps", 0))
    trainer.model.diagnostics_enabled = bool(trainer.diagnostic_every)
    trainer.data_collator.diagnostics_enabled = bool(trainer.diagnostic_every)
    if diagnostic_config.get("generation_probe"):
        from tvg_confidence.training_generation_probe import generation_probe_callback
        trainer.add_callback(generation_probe_callback(trainer, records, processor,
            PairedVideoCollator, video_messages, arm_dir, diagnostic_config["generation_probe"]))
    if full_soft:
        trainer.compact_checkpoint_contract = checkpoint_contract(manifest, sha256_file(ENTRY_PATH), world_size)
        trainer.add_callback(progress_callback(arm_dir, save_final=bool(contract.get("checkpoint_save_final", False))))
        resume_checkpoint = latest_checkpoint(arm_dir / "trainer-state", trainer.compact_checkpoint_contract)
        if trainer.is_world_process_zero():
            print(json.dumps({"stage": "full_soft_confidence_training_start", "protocol_version": manifest["protocol_version"], "fresh_base_initialization": manifest.get("generator_initialization") is None,
                "resume_checkpoint": resume_checkpoint, "target_optimizer_steps": train_dataset.target_optimizer_steps}), flush=True)
    trainer._diagnostic_window = (time.monotonic(), 0)
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.accelerator.wait_for_everyone()
    if int(result.global_step) != train_dataset.target_optimizer_steps:
        raise RuntimeError(
            "optimizer-step contract drift: "
            f"{result.global_step} != {train_dataset.target_optimizer_steps}"
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": str(manifest["protocol_version"]),
        "arm": arm,
        "queries": len(records),
        "micro_samples": len(train_dataset),
        "direct_micro_samples": len(records),
        "confidence_micro_samples": train_dataset.real_micro_samples - len(records),
        "confidence_supervision_mode": (
            "none"
            if manifest.get("confidence_supervision", {}).get("kind") == "none"
            else "inline_direct_target"
            if inline_confidence
            else "interleaved_replay"
        ),
        "zero_weight_padding_micro_samples": train_dataset.padding_micro_samples,
        "epochs": epochs,
        "query_passes": len(records) * epochs,
        "target_global_batch_queries": target_global_batch_queries,
        "target_optimizer_steps_per_epoch": (
            train_dataset.target_optimizer_steps_per_epoch
        ),
        "target_optimizer_steps": train_dataset.target_optimizer_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lm_head_checkpoint_chunk_tokens": LM_HEAD_CHECKPOINT_CHUNK_TOKENS,
        "nominal_global_query_equivalents_per_step": (
            world_size * gradient_accumulation_steps / micro_sample_ratio
        ),
        "objective_branch_scale": train_dataset.objective_branch_scale,
        "optimizer_steps": int(result.global_step),
        "resumed_from_checkpoint": resume_checkpoint,
        "lambda_conf": float(lambda_conf if arm == "joint" else 0.0),
        "lambda_exact_duplicate_unlikelihood": float(
            lambda_exact_duplicate if arm == "direct" else 0.0
        ),
        "endpoint_anchor": str(
            manifest["training_contract"].get("endpoint_anchor", "after_span")
        ),
        "head_initialization": manifest["head_initialization"] if arm == "joint" else None,
        "generator_initialization": manifest.get("generator_initialization"),
        "generator_update_mode": generator_update_mode,
        "generator_frozen": generator_frozen,
        "lm_loss_weight": 0.0 if generator_frozen else 1.0,
        "lora": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "targets": list(LORA_TARGETS),
        },
        "train_metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in result.metrics.items()
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "paper_main_eligible": False,
    }
    if trainer.is_world_process_zero():
        arm_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        adapter_dir = arm_dir / "generator-adapter"
        if generator_frozen:
            initialization = manifest.get("generator_initialization")
            if not isinstance(initialization, dict):
                raise RuntimeError(
                    "frozen_head_only requires a generator adapter initialization"
                )
            shutil.copytree(Path(initialization["adapter_dir"]), adapter_dir)
        else:
            unwrapped.generator.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(arm_dir / "processor")
        report["generator_adapter_files"] = {
            path.name: sha256_file(path)
            for path in sorted(adapter_dir.iterdir())
            if path.is_file()
        }
        if generator_frozen:
            expected_adapter = {
                "adapter_model.safetensors": str(
                    manifest["generator_initialization"]["adapter_model_sha256"]
                ),
                "adapter_config.json": str(
                    manifest["generator_initialization"]["adapter_config_sha256"]
                ),
            }
            observed_adapter = report["generator_adapter_files"]
            if any(
                observed_adapter.get(name) != expected_sha
                for name, expected_sha in expected_adapter.items()
            ):
                raise RuntimeError(
                    "frozen generator adapter is not byte-identical to initialization"
                )
        if arm == "joint":
            from safetensors.torch import save_file

            head_state = {
                name: value.detach().cpu().contiguous()
                for name, value in unwrapped.confidence_head.state_dict().items()
            }
            head_path = arm_dir / "confidence-head.safetensors"
            save_file(head_state, str(head_path))
            report["confidence_head_sha256"] = sha256_file(head_path)
            if manifest["head_initialization"].get("kind") == "random_span_hidden_binary_mlp":
                span_head_path = arm_dir / "span-confidence-head.safetensors"
                save_file(head_state, str(span_head_path))
                span_config_path = arm_dir / "span-head-config.json"
                write_json(
                    span_config_path,
                    {
                        "schema_version": 1,
                        "protocol_version": str(manifest["protocol_version"]),
                        "pooling": "mean over target interval text tokens",
                        "target": manifest["confidence_supervision"]["target"],
                        "ranking_weight": manifest["training_contract"].get("confidence_ranking_weight", 0.25),
                        "ranking_min_gap": 0.1,
                        "generator_confidence_stop_gradient": True,
                        "generator_identity": {
                            "adapter_model_sha256": report[
                                "generator_adapter_files"
                            ].get("adapter_model.safetensors"),
                            "adapter_config_sha256": report[
                                "generator_adapter_files"
                            ].get("adapter_config.json"),
                        },
                        "head_architecture": manifest["head_initialization"][
                            "architecture"
                        ],
                    },
                )
                report["span_confidence_head_sha256"] = sha256_file(
                    span_head_path
                )
                report["span_head_config_sha256"] = sha256_file(span_config_path)
        write_json(arm_dir / "report.json", report)
    trainer.accelerator.wait_for_everyone()
    del trainer, model, generator, confidence_head
    # The two arms run sequentially in one H20 job.  Force collection of any
    # Trainer/Accelerate reference cycles before constructing the second 4B
    # model; empty_cache alone cannot release still-referenced tensors.
    gc.collect()
    torch.cuda.empty_cache()
    return report


def load_completed_arm_for_recovery(
    output_root: Path,
    arm: str,
    *,
    protocol_version: str,
    expected_queries: int,
    expected_epochs: int,
    expected_steps: int,
) -> dict[str, Any] | None:
    """Accept completed arms or an interrupted arm's known runtime outputs.

    This entry point historically disabled Trainer checkpoints.  A GPU
    reschedule can therefore restart an interrupted arm, but it cannot recover
    its in-memory optimizer state.  Completed arms are immutable and may be
    skipped, while any partially published adapter/report is rejected instead
    of being silently overwritten.
    """

    arm_dir = output_root / arm
    if not arm_dir.exists():
        return None
    report_path = arm_dir / "report.json"
    entries = {path.name for path in arm_dir.iterdir()}
    if not report_path.is_file():
        allowed = {"trainer-state"}
        if protocol_version in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS:
            allowed.update({
                "training-progress.json", "training-metrics.jsonl",
                "training-diagnostics-latest.json", "train-generation-probe",
            })
        unexpected = entries - allowed
        if unexpected:
            raise SystemExit(
                f"partial {arm} publication is not recoverable: {sorted(unexpected)}"
            )
        trainer_state = arm_dir / "trainer-state"
        if trainer_state.exists() and any(trainer_state.iterdir()) and protocol_version not in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS:
            raise SystemExit(
                f"unexpected checkpoint state for checkpoint-free {arm} arm"
            )
        return None

    report = json.loads(report_path.read_text())
    checks = {
        "protocol": report.get("protocol_version") == protocol_version,
        "arm": report.get("arm") == arm,
        "queries": report.get("queries") == expected_queries,
        "epochs": report.get("epochs") == expected_epochs,
        "optimizer_steps": report.get("optimizer_steps") == expected_steps,
        "target_optimizer_steps": report.get("target_optimizer_steps")
        == expected_steps,
    }
    adapter_dir = arm_dir / "generator-adapter"
    adapter_files = report.get("generator_adapter_files")
    checks["adapter_manifest"] = isinstance(adapter_files, dict) and bool(
        adapter_files
    )
    if isinstance(adapter_files, dict):
        checks["adapter_files"] = all(
            (adapter_dir / name).is_file()
            and sha256_file(adapter_dir / name) == expected_sha
            for name, expected_sha in adapter_files.items()
        )
    else:
        checks["adapter_files"] = False
    if arm == "joint":
        head_path = arm_dir / "confidence-head.safetensors"
        checks["confidence_head"] = (
            head_path.is_file()
            and sha256_file(head_path) == report.get("confidence_head_sha256")
        )
    if not all(checks.values()):
        raise SystemExit(f"completed {arm} recovery gate failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-conf", type=float, default=0.25)
    parser.add_argument(
        "--run-mode", choices=("both", "direct-only", "joint-only"), default="both"
    )
    parser.add_argument("--reference-direct-report", type=Path)
    parser.add_argument("--expected-reference-direct-report-sha256")
    parser.add_argument(
        "--recover-partial-output",
        action="store_true",
        help=(
            "Permit a matching interrupted output root. Completed arms are "
            "strictly validated and skipped; an incomplete checkpoint-free arm "
            "restarts from its frozen initialization."
        ),
    )
    args = parser.parse_args()

    # GPU H20 ranks previously performed all shared filesystem manifest/data reads
    # before the first NCCL call.  A delayed or failed reader then left the
    # remaining ranks looking like an NCCL bootstrap hang.  The successful
    # TimeLens2 head jobs establish the communicator first, so r16 can request
    # the same ordering without changing the scientific objective.
    training_distributed_backend = os.environ.get(
        "TVG_TRAIN_DISTRIBUTED_BACKEND", "nccl"
    ).lower()
    if training_distributed_backend not in {"gloo", "nccl"}:
        raise SystemExit(
            "TVG_TRAIN_DISTRIBUTED_BACKEND must be gloo or nccl"
        )
    early_distributed_init = os.environ.get("TVG_EARLY_DISTRIBUTED_INIT") == "1"
    if early_distributed_init:
        if os.environ.get("TVG_SINGLE_RANK_FALLBACK") == "1":
            raise SystemExit(
                "TVG_EARLY_DISTRIBUTED_INIT is incompatible with the single-rank fallback"
            )
        import torch
        import torch.distributed as distributed

        early_backend = os.environ.get(
            "TVG_EARLY_DISTRIBUTED_BACKEND", "nccl"
        ).lower()
        if early_backend not in {"gloo", "nccl"}:
            raise SystemExit(
                "TVG_EARLY_DISTRIBUTED_BACKEND must be gloo or nccl"
            )
        early_rank = int(os.environ.get("RANK", "0"))
        early_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        early_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if early_world_size <= 1 or not torch.cuda.is_available():
            raise SystemExit(
                "early distributed initialization requires a multi-GPU CUDA launcher"
            )
        if early_backend == "nccl":
            torch.cuda.set_device(early_local_rank)
        if early_rank == 0:
            print(
                json.dumps(
                    {
                        "stage": "early_distributed_init_start",
                        "backend": early_backend,
                        "world_size": early_world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not distributed.is_initialized():
            distributed.init_process_group(backend=early_backend)
        early_collective = torch.ones(
            1,
            dtype=torch.float32,
            device=(
                torch.device("cuda", early_local_rank)
                if early_backend == "nccl"
                else torch.device("cpu")
            ),
        )
        distributed.all_reduce(early_collective, op=distributed.ReduceOp.SUM)
        if early_backend == "nccl":
            torch.cuda.synchronize(early_local_rank)
        early_collective_value = float(early_collective.item())
        if early_collective_value != float(early_world_size):
            raise RuntimeError(
                "early NCCL collective returned an invalid value: "
                f"{early_collective_value} != {early_world_size}"
            )
        del early_collective
        if early_rank == 0:
            print(
                json.dumps(
                    {
                        "stage": "early_distributed_init_complete",
                        "backend": early_backend,
                        "value": early_collective_value,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if early_backend == "gloo" and training_distributed_backend != "gloo":
            # The early group is only a rank-readiness gate.  Tear it down so
            # the actual DDP group can follow the proven H20 ordering: load the
            # model first, then initialize NCCL as the successful head jobs do.
            distributed.barrier()
            distributed.destroy_process_group()
            if early_rank == 0:
                print(
                    json.dumps(
                        {
                            "stage": "early_distributed_group_destroyed",
                            "backend": early_backend,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        elif early_backend == training_distributed_backend and early_rank == 0:
            print(
                json.dumps(
                    {
                        "stage": "early_distributed_group_reused",
                        "backend": early_backend,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    manifest = load_manifest(args.input_manifest, args.expected_manifest_sha256)
    protocol_version = str(manifest["protocol_version"])
    expected_lambda_conf = (
        0.0
        if protocol_version
        in {
            POSITIVE_ONLY_FIXED16_HUNGARIAN_CONFIDENCE_PROTOCOL,
            POSITIVE_ONLY_FIXED32_ALL_GT_RANDOM_NEGATIVE_CONFIDENCE_PROTOCOL,
            POSITIVE_ONLY_FIXED32_GT_NMS_ANCHOR_COORDINATE_PROTOCOL,
            BOUNDED_COVERAGE_POOL_PROTOCOL,
        }
        else 0.25
    )
    if not math.isclose(
        args.lambda_conf, expected_lambda_conf, abs_tol=1e-12
    ):
        raise SystemExit(
            f"{protocol_version} freezes lambda_conf={expected_lambda_conf}"
        )
    output_was_nonempty = args.output_dir.exists() and any(
        args.output_dir.iterdir()
    )
    if output_was_nonempty and not args.recover_partial_output:
        raise SystemExit(f"refusing non-empty output directory: {args.output_dir}")

    records, data_summary = prepare_records(manifest)
    recovered_direct_report = None
    recovered_joint_report = None
    if output_was_nonempty:
        allowed_root_entries = {"preflight.json", "direct", "joint"}
        root_entries = {path.name for path in args.output_dir.iterdir()}
        if "_SUCCESS" in root_entries:
            if not (args.output_dir / "report.json").is_file():
                raise SystemExit("successful output is missing report.json")
            print(
                json.dumps(
                    {
                        "stage": "preemption_recovery_already_complete",
                        "output_dir": str(args.output_dir),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        unexpected_root = root_entries - allowed_root_entries
        if unexpected_root:
            raise SystemExit(
                "partial output has unexpected root entries: "
                f"{sorted(unexpected_root)}"
            )
        preflight_path = args.output_dir / "preflight.json"
        if not preflight_path.is_file():
            raise SystemExit("partial recovery requires preflight.json")
        existing_preflight = json.loads(preflight_path.read_text())
        preflight_checks = {
            "protocol": existing_preflight.get("protocol_version")
            == manifest["protocol_version"],
            "manifest_sha": existing_preflight.get("manifest_sha256")
            == args.expected_manifest_sha256,
            "status": existing_preflight.get("status") == "training",
            "paper_main_eligible": existing_preflight.get("paper_main_eligible")
            is False,
        }
        if not all(preflight_checks.values()):
            raise SystemExit(
                f"partial recovery preflight mismatch: {preflight_checks}"
            )
        contract = manifest["training_contract"]
        recovery_epochs = int(contract["epochs_per_arm"])
        recovery_steps = math.ceil(
            len(records) / int(contract["target_global_batch_queries"])
        ) * recovery_epochs
        if args.run_mode in {"both", "direct-only"}:
            recovered_direct_report = load_completed_arm_for_recovery(
                args.output_dir,
                "direct",
                protocol_version=str(manifest["protocol_version"]),
                expected_queries=len(records),
                expected_epochs=recovery_epochs,
                expected_steps=recovery_steps,
            )
        if args.run_mode in {"both", "joint-only"}:
            recovered_joint_report = load_completed_arm_for_recovery(
                args.output_dir,
                "joint",
                protocol_version=str(manifest["protocol_version"]),
                expected_queries=len(records),
                expected_epochs=recovery_epochs,
                expected_steps=recovery_steps,
            )
        print(
            json.dumps(
                {
                    "stage": "preemption_recovery_validated",
                    "completed_direct": recovered_direct_report is not None,
                    "completed_joint": recovered_joint_report is not None,
                    "incomplete_arm_policy": "resume_verified_compact_checkpoint" if protocol_version in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS else "restart_from_frozen_initialization",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    generator_update_mode = str(
        manifest.get("training_contract", {}).get(
            "generator_update_mode", "lm_only_with_detached_confidence"
        )
    )
    if generator_update_mode == "frozen_head_only" and args.run_mode != "joint-only":
        raise SystemExit("frozen_head_only requires --run-mode joint-only")
    direct_reference = None
    standalone_soft_joint = (
        (protocol_version == BOUNDED_POOL_SOFT_CONFIDENCE_PROTOCOL and manifest["training_contract"].get("standalone_joint_pilot") is True)
        or (protocol_version in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS and manifest["training_contract"].get("standalone_joint_full") is True)
    )
    if standalone_soft_joint:
        if (
            args.run_mode != "joint-only"
            or args.reference_direct_report is not None
            or args.expected_reference_direct_report_sha256 is not None
            or manifest["confidence_supervision"].get("kind") not in {"inline_verifier_pyes", "inline_gt_binary"}
            or manifest["training_contract"].get("confidence_supervision_mode") != "detached_replay_target"
            or manifest["training_contract"].get("confidence_backprop_to_generator") is not False
            or manifest["head_initialization"].get("kind") != "random_span_hidden_binary_mlp"
            or generator_update_mode != "lm_only_with_detached_confidence"
        ):
            raise SystemExit("standalone soft-confidence training contract drift")
        if protocol_version in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS and (
            manifest.get("generator_initialization") is not None
            or manifest.get("base_initialization_contract", {}).get("fresh_lora") is not True
            or manifest["model"]["revision"] != "ddbb6cb944f13ce21e59e85da23c5f356107260e"
        ):
            raise SystemExit("full soft-confidence training must start from original TL2-4B with a fresh LoRA")
    elif args.run_mode == "joint-only":
        if (
            args.reference_direct_report is None
            or args.expected_reference_direct_report_sha256 is None
        ):
            raise SystemExit("joint-only requires the completed direct report")
        if (
            sha256_file(args.reference_direct_report)
            != args.expected_reference_direct_report_sha256
        ):
            raise SystemExit("reference direct-report SHA mismatch")
        direct_reference = json.loads(args.reference_direct_report.read_text())
        reference_epochs = int(manifest["training_contract"]["epochs_per_arm"])
        reference_steps = (
            math.ceil(
                len(records)
                / int(manifest["training_contract"]["target_global_batch_queries"])
            )
            * reference_epochs
        )
        direct_checks = {
            "arm": direct_reference.get("arm") == "direct",
            "queries": direct_reference.get("queries") == len(records),
            "epochs": direct_reference.get("epochs") == reference_epochs,
            "optimizer_steps": direct_reference.get("optimizer_steps")
            == direct_reference.get("target_optimizer_steps")
            == reference_steps,
            "endpoint_anchor": (
                direct_reference.get("endpoint_anchor") in {"after_span", "after_end"}
                if manifest.get("protocol_version")
                == POSITIVE_ONLY_VARIABLE_CLOSE_DETACHED_CONFIDENCE_PROTOCOL
                else direct_reference.get("endpoint_anchor") == "after_end"
            ),
        }
        if not all(direct_checks.values()):
            raise SystemExit(f"reference direct-report gate failed: {direct_checks}")
        if generator_update_mode == "frozen_head_only":
            initialization = manifest.get("generator_initialization")
            if not isinstance(initialization, dict):
                raise SystemExit(
                    "frozen_head_only requires the direct generator initialization"
                )
            frozen_checks = {
                "confidence_stop_gradient": manifest["training_contract"].get(
                    "confidence_backprop_to_generator"
                )
                is False,
                "adapter_model_matches_direct": initialization.get(
                    "adapter_model_sha256"
                )
                == direct_reference.get("generator_adapter_files", {}).get(
                    "adapter_model.safetensors"
                ),
                "adapter_config_matches_direct": initialization.get(
                    "adapter_config_sha256"
                )
                == direct_reference.get("generator_adapter_files", {}).get(
                    "adapter_config.json"
                ),
            }
            if not all(frozen_checks.values()):
                raise SystemExit(
                    f"frozen head-only generator gate failed: {frozen_checks}"
                )
    elif (
        args.reference_direct_report is not None
        or args.expected_reference_direct_report_sha256 is not None
    ):
        raise SystemExit("direct reference is only valid in joint-only mode")

    # In the rank-0 fallback, leaving all eight allocated devices visible makes
    # TrainingArguments silently enable legacy DataParallel.  Its effective
    # batch then becomes eight and violates the frozen one-query collator.  Hide
    # the seven idle devices before torch first initializes CUDA; ranks 1--7
    # only wait on completion markers and never select a device.
    if os.environ.get("TVG_SINGLE_RANK_FALLBACK") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "TVG_SINGLE_RANK_CUDA_VISIBLE_DEVICES", "0"
        )

    import torch
    import transformers
    from transformers import AutoProcessor

    if transformers.__version__ != "4.57.1":
        raise SystemExit(
            "v1 freezes transformers==4.57.1; observed "
            f"{transformers.__version__}"
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    contract = manifest["training_contract"]
    expected_world_size = int(contract.get("world_size", 8))
    expected_local_world_size = int(contract.get("local_world_size", 8))
    if (
        world_size != expected_world_size
        or local_world_size != expected_local_world_size
        or not torch.cuda.is_available()
    ):
        raise SystemExit(
            "distributed contract mismatch: "
            f"world={world_size}/{expected_world_size}, "
            f"local={local_world_size}/{expected_local_world_size}"
        )
    allocated_world_size = world_size
    allocated_local_world_size = local_world_size
    single_rank_fallback = os.environ.get("TVG_SINGLE_RANK_FALLBACK") == "1"
    if single_rank_fallback and rank != 0:
        print(
            json.dumps(
                {
                    "stage": "single_rank_fallback_worker_wait",
                    "allocated_rank": rank,
                    "rank0_success_marker": str(args.output_dir / "_SUCCESS"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        heartbeat = 0
        while True:
            if (args.output_dir / "_SUCCESS").is_file():
                print(
                    json.dumps(
                        {"stage": "single_rank_fallback_worker_complete", "rank": rank},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return 0
            failure_marker = args.output_dir / "_SINGLE_RANK_FAILED"
            if failure_marker.is_file():
                raise RuntimeError(failure_marker.read_text(errors="replace"))
            time.sleep(30)
            heartbeat += 1
            if heartbeat % 20 == 0:
                print(
                    json.dumps(
                        {"stage": "single_rank_fallback_worker_wait", "rank": rank},
                        sort_keys=True,
                    ),
                    flush=True,
                )

    torch.cuda.set_device(local_rank)
    if single_rank_fallback:
        # The H20 allocation still contains eight processes, but repeated
        # pre-model one-element collectives proved that this container's NCCL
        # communicator cannot initialize.  Rank 0 therefore performs the same
        # frozen objective with 8x gradient accumulation while ranks 1--7 wait.
        rank = 0
        local_rank = 0
        world_size = 1
        local_world_size = 1
        # WORLD_SIZE=1 together with LOCAL_RANK=0 still makes Accelerate create
        # a one-rank NCCL process group.  This H20 runtime stalls even in that
        # communicator.  Preserve the effective batch contract separately and
        # remove the torchrun discovery variables before TrainingArguments is
        # constructed so rank 0 uses ordinary single-process CUDA.
        os.environ["TVG_EFFECTIVE_WORLD_SIZE"] = "1"
        for distributed_name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            os.environ.pop(distributed_name, None)
        print(
            json.dumps(
                {
                    "stage": "single_rank_fallback_active",
                    "allocated_world_size": allocated_world_size,
                    "effective_world_size": world_size,
                    "distributed_launcher_environment_cleared": True,
                    "scientific_protocol_changed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        # Match the initialization order of the H20 action-head job that already
        # completed successfully on this image.  Letting Trainer initialize NCCL
        # only after every rank has built the model repeatedly stalled in NCCL
        # bootstrap; establish the process group first and let Accelerate reuse it.
        import torch.distributed as distributed

        if not distributed.is_initialized():
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "stage": "distributed_init_start",
                            "backend": training_distributed_backend,
                            "world_size": world_size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            distributed.init_process_group(backend=training_distributed_backend)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "stage": "distributed_init_complete",
                            "backend": training_distributed_backend,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        # init_process_group is lazy for NCCL; force communicator creation before
        # the 4B model is loaded so transport failures remain an early signal.
        if rank == 0:
            print(
                json.dumps(
                    {
                        "stage": "distributed_collective_warmup_start",
                        "backend": training_distributed_backend,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        collective_warmup = torch.ones(
            1,
            dtype=torch.float32,
            device=torch.device("cuda", local_rank),
        )
        distributed.all_reduce(collective_warmup, op=distributed.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        collective_warmup_value = float(collective_warmup.item())
        if collective_warmup_value != float(world_size):
            raise RuntimeError(
                "distributed collective warmup returned an invalid value: "
                f"{collective_warmup_value} != {world_size}"
            )
        del collective_warmup
        if rank == 0:
            print(
                json.dumps(
                    {
                        "stage": "distributed_collective_warmup_complete",
                        "backend": training_distributed_backend,
                        "value": collective_warmup_value,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    video_cache_preflight = None
    if os.environ.get("TVG_BOUNDED_VIDEO_CACHE_PREFLIGHT") == "1":
        from tvg_confidence.bounded_video_cache import distributed_cache_preflight
        video_cache_preflight = distributed_cache_preflight(
            os.environ.get("TVG_VIDEO_CACHE_PROBE_ROOT", "/tmp/tvg_video_cache_probe"),
            args.output_dir / "joint/trainer-state/video-cache-preflight.json",
            rank, world_size,
        )
    model_source_path = str(manifest["model"]["path"])
    model_revision = str(manifest["model"]["revision"])
    runtime_model_path = localize_model_directory(
        model_source_path,
        model_revision,
        rank,
    )
    manifest["model"] = {
        **manifest["model"],
        "path": str(runtime_model_path),
        "source_path": model_source_path,
    }
    processor = AutoProcessor.from_pretrained(
        str(runtime_model_path),
        revision=model_revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    # The training bundle vendors decord for GPU as well as GPU.  Unlike
    # torchvision.read_video, decord samples the requested frame indices without
    # materializing every decoded frame of a long source video in host memory.
    video_reader_backend = os.environ.get(
        "TVG_VIDEO_READER_BACKEND", "decord"
    )
    if video_reader_backend not in {"decord", "torchvision", "torchcodec"}:
        raise SystemExit(
            f"unsupported TVG_VIDEO_READER_BACKEND={video_reader_backend}"
        )
    os.environ["FORCE_QWENVL_VIDEO_READER"] = video_reader_backend
    video_reader_version = None
    if video_reader_backend == "decord":
        import decord

        video_reader_version = str(decord.__version__)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        preflight = {
            "schema_version": 1,
            "protocol_version": str(manifest["protocol_version"]),
            "status": "training",
            "manifest_sha256": args.expected_manifest_sha256,
            "data": data_summary,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "world_size": world_size,
                "local_world_size": local_world_size,
                "allocated_world_size": allocated_world_size,
                "allocated_local_world_size": allocated_local_world_size,
                "single_rank_fallback": single_rank_fallback,
                "early_distributed_init": early_distributed_init,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "model_source_path": model_source_path,
                "runtime_model_path": str(runtime_model_path),
                "model_localized": str(runtime_model_path) != model_source_path,
                "video_reader_backend": video_reader_backend,
                "video_reader_version": video_reader_version,
                "video_cache_preflight": video_cache_preflight,
                "video_cache_max_bytes": int(os.environ.get("TVG_VIDEO_CACHE_MAX_BYTES", str(8 * 1024**3))),
                "generator_initialization": manifest.get("generator_initialization"),
                "generator_update_mode": generator_update_mode,
            },
            "fairness_contract": {
                "same_initial_checkpoint": None if standalone_soft_joint else True,
                "same_6500_query_set": None if standalone_soft_joint else True,
                "same_epochs_per_arm": int(contract["epochs_per_arm"]),
                "query_order": "seeded_arm_specific_micro_sample_permutation",
                "same_gt_generation_branch": None if standalone_soft_joint else True,
                "same_lora_parameterization": None if standalone_soft_joint else True,
                "rollout_language_model_loss": False,
                "joint_additional_training_compute_disclosed": True,
                "joint_only_objective_difference": (
                    "not a controlled comparison; bounded pool plus legacy soft replay"
                    if standalone_soft_joint else "endpoint confidence supervision"
                ),
                "standalone_joint_pilot": standalone_soft_joint and protocol_version == BOUNDED_POOL_SOFT_CONFIDENCE_PROTOCOL,
                "standalone_joint_full": protocol_version in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS,
                "generator_frozen": generator_update_mode == "frozen_head_only",
                "lm_loss_weight": (
                    0.0 if generator_update_mode == "frozen_head_only" else 1.0
                ),
                "run_mode": args.run_mode,
                "reference_direct_report_sha256": (
                    args.expected_reference_direct_report_sha256
                    if args.run_mode == "joint-only"
                    else None
                ),
            },
            "paper_main_eligible": False,
        }
        if not (args.output_dir / "preflight.json").is_file():
            write_json(args.output_dir / "preflight.json", preflight)

    direct_report = recovered_direct_report
    if args.run_mode in {"both", "direct-only"} and direct_report is None:
        direct_report = train_arm(
            arm="direct",
            records=records,
            manifest=manifest,
            processor=processor,
            output_root=args.output_dir,
            lambda_conf=args.lambda_conf,
        )
    joint_report = recovered_joint_report
    if args.run_mode in {"both", "joint-only"} and joint_report is None:
        joint_report = train_arm(
            arm="joint",
            records=records,
            manifest=manifest,
            processor=processor,
            output_root=args.output_dir,
            lambda_conf=args.lambda_conf,
        )

    if rank == 0:
        final_report = {
            "schema_version": 1,
            "protocol_version": str(manifest["protocol_version"]),
            "decision": (
                "DIRECT_AND_JOINT_SINGLE_STAGE_SFT_COMPLETED"
                if args.run_mode == "both"
                else (
                    (
                        "DIRECT_EXPANDED_GT_ONLY_SFT_CONTROL_COMPLETED"
                        if manifest["protocol_version"]
                        in {
                            EXPANDED_GT_SFT_CONTROL_PROTOCOL,
                            REVISED_MULTISPAN_GT_ONLY_PROTOCOL,
                            EXPANDED_RESIDUAL_COMPLETION_PROTOCOL,
                        }
                        else "DIRECT_EXPLICIT_CANDIDATE_CONFIDENCE_SFT_COMPLETED"
                    )
                    if args.run_mode == "direct-only"
                    else (
                        "BOUNDED_POOL_SOFT_CONFIDENCE_FULL_COMPLETED"
                        if protocol_version in BOUNDED_POOL_SOFT_CONFIDENCE_FULL_PROTOCOLS
                        else "BOUNDED_POOL_SOFT_CONFIDENCE_PILOT_COMPLETED"
                        if standalone_soft_joint
                        else "FROZEN_GENERATOR_CONFIDENCE_HEAD_ONLY_COMPLETED"
                        if generator_update_mode == "frozen_head_only"
                        else "JOINT_SINGLE_STAGE_SFT_COMPLETED_WITH_FROZEN_DIRECT_REFERENCE"
                    )
                )
            ),
            "data": data_summary,
            "direct": direct_report,
            "direct_reference": (
                {
                    "path": str(args.reference_direct_report),
                    "sha256": args.expected_reference_direct_report_sha256,
                    "optimizer_steps": direct_reference["optimizer_steps"],
                    "train_loss": direct_reference["train_metrics"]["train_loss"],
                }
                if direct_reference is not None
                else None
            ),
            "joint": joint_report,
            "comparison_status": "training_complete; independent inference required",
            "paper_main_eligible": False,
        }
        write_json(args.output_dir / "report.json", final_report)
        write_json(
            args.output_dir / "_SUCCESS",
            {
                "protocol_version": str(manifest["protocol_version"]),
                "report_sha256": sha256_file(args.output_dir / "report.json"),
            },
        )
        print(json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exc:
        if (
            os.environ.get("TVG_SINGLE_RANK_FALLBACK") == "1"
            and os.environ.get("RANK", "0") == "0"
            and "--output-dir" in sys.argv
        ):
            output_index = sys.argv.index("--output-dir") + 1
            if output_index < len(sys.argv):
                failure_root = Path(sys.argv[output_index])
                failure_root.mkdir(parents=True, exist_ok=True)
                write_json(
                    failure_root / "_SINGLE_RANK_FAILED",
                    {
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
        raise
    raise SystemExit(exit_code)
