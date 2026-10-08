"""GT-only supervision for the paper SFT confidence-replay branch."""
from __future__ import annotations

import json
import math

PROTOCOL = "timelens2_bounded_pool_gt_binary_sft_v1"


def replay_binary_targets(source: dict) -> list[float]:
    """Label unchanged replay spans against the full, duration-aligned TRAIN GT."""
    gt = source["gt_intervals"]
    candidates = json.loads(source["detached_confidence_replay_target"])
    slots = source["detached_confidence_replay_candidate_slots"]
    if not gt or len(candidates) != len(slots):
        raise ValueError("GT/replay candidate identity is incomplete")
    labels = []
    for interval, slot in zip(candidates, slots):
        start, end = interval
        if not all(math.isfinite(x) for x in interval) or end <= start:
            raise ValueError("invalid confidence-replay interval")
        left, right = slot["span_char_span"]
        if json.loads(source["detached_confidence_replay_target"][left:right]) != interval:
            raise ValueError("replay interval and hidden-state span disagree")
        overlap = max(
            max(0.0, min(end, b) - max(start, a))
            / (max(end, b) - min(start, a))
            for a, b in gt
        )
        labels.append(float(overlap >= 0.5))
    return labels


def validate_contract(manifest: dict) -> None:
    if manifest["protocol_version"] != PROTOCOL:
        return
    spec = manifest["confidence_supervision"]
    contract = manifest["training_contract"]
    if not (
        spec["kind"] == "inline_gt_binary"
        and spec["target"] == "max_gt_tiou_ge_0.5"
        and spec["hard_confidence_labels"] is True
        and not any("teacher" in k or "verifier" in k for k in spec)
        and contract["confidence_ranking_weight"] == 0
        and contract["confidence_backprop_to_generator"] is False
        and manifest["generator_initialization"] is None
        and manifest["head_initialization"]["kind"] == "random_span_hidden_binary_mlp"
    ):
        raise ValueError("GT-only BCE experiment contract drift")
