"""Skip left padding and prompt vocabulary projection, with an actual actor parity gate."""
import json
import os
from pathlib import Path
import time

import torch


def trim_micro_batch(inputs):
    """Remove only a prefix masked in every row; preserve response and mRoPE values."""
    mask = inputs["attention_mask"]
    width = inputs["input_ids"].shape[-1]
    response_width = inputs["responses"].shape[-1]
    if mask.ndim != 2 or mask.shape != inputs["input_ids"].shape:
        raise RuntimeError("unexpected actor attention layout")
    occupied = mask.bool().any(dim=0).nonzero(as_tuple=False)
    if not len(occupied):
        raise RuntimeError("empty actor prompt")
    cut = int(occupied[0, 0])
    if cut >= width - response_width:
        raise RuntimeError("left-padding trim would remove the entire prompt")
    for key in ("input_ids", "attention_mask", "position_ids"):
        if inputs[key].shape[-1] != width:
            raise RuntimeError("actor position/token alignment changed")
        if cut:
            inputs[key] = inputs[key][..., cut:].contiguous()
    return width, width - cut


def forward_micro_batch(actor, inputs, temperature, calculate_entropy):
    # The original frozen/online parents used full projection, without trimming.
    # Enable these historical continuation optimizations only for the L phase.
    if os.environ.get("TVG_RL_ACTOR_PERFORMANCE", "0") != "1":
        actor._tvg_optimized = False
        return actor._forward_micro_batch_impl(inputs, temperature, calculate_entropy)
    if getattr(actor, "_tvg_speed_checked", False):
        return actor._forward_micro_batch_impl(inputs, temperature, calculate_entropy)
    if torch.is_grad_enabled() or actor.tvg_scorer.collect_training:
        raise RuntimeError("actual actor parity must pass in validation before training")
    # Both forwards use the restored actor on each actual FSDP rank. No optimizer
    # or head updates occur, and the surrounding generator RNG is restored.
    device = inputs["input_ids"].device
    response_width = inputs["responses"].shape[-1]
    valid = inputs["attention_mask"][:, -response_width:].bool()
    before = dict(inputs)
    with torch.random.fork_rng(devices=[device]):
        actor._tvg_optimized = False
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        old = actor._forward_micro_batch_impl(before, temperature, calculate_entropy)
        old_hidden = actor.tvg_scorer.hidden[:, -response_width:].float().clone()
        old_scores = actor.tvg_scorer.score(before)
        torch.cuda.synchronize(device)
        old_seconds = time.perf_counter() - started
        actor._tvg_optimized = True
        started = time.perf_counter()
        new = actor._forward_micro_batch_impl(inputs, temperature, calculate_entropy)
        captured = actor.tvg_scorer.hidden
        new_hidden = captured[:, -response_width:].float()
        new_scores = actor.tvg_scorer.score(inputs)
        # The ordinary caller still performs its one score/collection operation.
        actor.tvg_scorer.hidden = captured
        torch.cuda.synchronize(device)
        new_seconds = time.perf_counter() - started
        log_diff = (old[1].float() - new[1].float())[valid].abs()
        hidden_diff = (old_hidden[valid] - new_hidden[valid]).square().mean().sqrt()
        hidden_scale = old_hidden[valid].square().mean().sqrt().clamp_min(1e-8)
        score_mask = old_scores >= 0
        if not torch.equal(score_mask, new_scores >= 0):
            raise RuntimeError("optimized actor changed candidate/head alignment")
        score_diff = (old_scores[score_mask] - new_scores[score_mask]).abs()
        report = {"status": "passed", "rank": torch.distributed.get_rank(),
            "untrimmed_length": before["input_ids"].shape[-1],
            "trimmed_length": inputs["input_ids"].shape[-1],
            "projection_positions_before": before["input_ids"].shape[-1],
            "projection_positions_after": response_width + 1,
            "valid_response_tokens": int(valid.sum()), "candidate_scores": int(score_mask.sum()),
            "log_prob_max_abs": float(log_diff.max()), "log_prob_mean_abs": float(log_diff.mean()),
            "response_hidden_relative_rmse": float(hidden_diff / hidden_scale),
            "head_score_max_abs": float(score_diff.max()) if score_diff.numel() else 0.0,
            "baseline_forward_and_head_seconds": old_seconds,
            "optimized_forward_and_head_seconds": new_seconds,
            "generator_or_head_optimizer_steps": 0,
            "scope": "actual restored FSDP actor validation forward; one microbatch, not steady-state throughput"}
        if (report["log_prob_max_abs"] > 0.1 or report["log_prob_mean_abs"] > 0.01
                or report["response_hidden_relative_rmse"] > 0.005 or report["head_score_max_abs"] > 0.005):
            report["status"] = "failed"
        output = Path(os.environ["TVG_RL_OUTPUT"])
        (output / ("actor-performance-parity-rank" + str(report["rank"]) + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"stage": "actual_actor_performance_parity", **report}), flush=True)
        if report["status"] != "passed":
            raise RuntimeError("response-only/left-trim actor parity failed")
    actor._tvg_speed_checked = True
    return new


def record_shape(actor, original, trimmed, projected):
    stats = getattr(actor, "tvg_shape_stats", None)
    if stats is None:
        return
    stats["microbatches"] += 1
    stats["input_positions_before"] += original
    stats["input_positions_after"] += trimmed
    stats["projected_positions"] += projected


def shape_metrics(stats, prefix):
    n = max(1, stats["microbatches"])
    return {"perf/" + prefix + "_" + key + ("_mean" if key != "microbatches" else ""): value / n if key != "microbatches" else value
            for key, value in stats.items()}


def reset_stats(actor):
    actor.tvg_shape_stats = dict(microbatches=0, input_positions_before=0,
        input_positions_after=0, projected_positions=0)
