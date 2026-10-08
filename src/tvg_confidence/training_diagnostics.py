"""Detached, count-weighted train diagnostics; these are not held-out metrics."""
from __future__ import annotations

import math
import time

GROUPS = ("gt", "alternative", "format")


def coordinate_statistics(logits, labels, losses, weights, groups):
    import torch

    with torch.no_grad():
        correct = logits.argmax(-1).eq(labels)
        return torch.stack([
            value
            for group in range(3)
            for value in (
                (losses.detach() * (groups == group)).sum(),
                (groups == group).sum().float(),
                (correct & (groups == group)).sum().float(),
                (losses.detach() * weights * (groups == group)).sum(),
            )
        ])


def confidence_statistics(logits, targets, better, worse):
    import torch

    with torch.no_grad():
        predictions = logits.detach().sigmoid()
        delta = predictions - targets
        return {
            "conf_candidates": predictions.new_tensor(predictions.numel()),
            "conf_prediction_sum": predictions.sum(),
            "conf_prediction_square_sum": predictions.square().sum(),
            "conf_teacher_sum": targets.sum(),
            "conf_teacher_square_sum": targets.square().sum(),
            "conf_absolute_error_sum": delta.abs().sum(),
            "conf_squared_error_sum": delta.square().sum(),
            "conf_saturated_sum": ((predictions < .05) | (predictions > .95)).sum().float(),
            "conf_pairs": predictions.new_tensor(better.numel()),
            "conf_pair_correct_sum": (logits.detach()[better] > logits.detach()[worse]).sum().float(),
        }


def summarize(values):
    """All input values are sums with explicit denominators, never batch averages."""
    result = dict(values)

    def ratio(name, numerator, denominator):
        if values.get(denominator, 0) > 0:
            result[name] = values.get(numerator, 0) / values[denominator]

    for group in GROUPS:
        ratio(f"ce_{group}", f"ce_{group}_sum", f"tokens_{group}")
        ratio(f"token_accuracy_{group}", f"correct_{group}", f"tokens_{group}")
        ratio(f"weighted_ce_{group}_contribution", f"weighted_ce_{group}_sum", "direct_queries")
    for name in ("lm_loss", "confidence_loss", "soft_bce", "ranking_loss"):
        ratio(name, name + "_sum", "direct_queries" if name == "lm_loss" else "confidence_queries")
    for name in ("objective_lm", "objective_confidence"):
        ratio(name, name + "_sum", "micro_samples")
    for name, numerator in (("confidence_soft_mae", "conf_absolute_error_sum"),
                            ("confidence_soft_mse", "conf_squared_error_sum"),
                            ("confidence_saturation_fraction", "conf_saturated_sum")):
        ratio(name, numerator, "conf_candidates")
    ratio("confidence_pair_accuracy_gap01", "conf_pair_correct_sum", "conf_pairs")
    for kind in ("prediction", "teacher"):
        n = values.get("conf_candidates", 0)
        if n:
            mean = values[f"conf_{kind}_sum"] / n
            result[f"confidence_{kind}_mean"] = mean
            result[f"confidence_{kind}_std"] = math.sqrt(max(0, values[f"conf_{kind}_square_sum"] / n - mean**2))
    ratio("cache_hit_rate", "cache_hit", "micro_samples")
    ratio("target_candidates_per_direct_query", "target_slots", "direct_queries")
    ratio("gt_events_per_direct_query", "gt_events", "direct_queries")
    # Runtime sums cover ranks; the denominator makes these per-micro-input values.
    for key in ("cache_localize_s", "cache_lock_wait_s", "cache_copy_s", "decode_s", "preprocess_s", "collate_s"):
        ratio(key + "_per_micro", key, "micro_samples")
    return result


class TrainingDiagnosticsMixin:
    """Trainer hook: no per-token CPU synchronization or logging from recompute."""
    diagnostic_every = 0

    def training_step(self, *args, **kwargs):
        started = time.monotonic()
        result = super().training_step(*args, **kwargs)
        if self.diagnostic_every:
            self._diagnostic_sums["forward_backward_host_wall_s"] = self._diagnostic_sums.get("forward_backward_host_wall_s", 0) + time.monotonic() - started
        return result

    def add_diagnostics(self, inputs, outputs):
        if not self.diagnostic_every:
            return
        metadata = inputs.get("_training_diagnostics", {})
        if not metadata.get("micro_samples", 0):
            return  # Zero-weight padding must not affect quality denominators.
        if not hasattr(self, "_diagnostic_sums"):
            self._diagnostic_sums = {}
        for key, value in {**metadata, **outputs.get("diagnostics", {})}.items():
            if hasattr(value, "detach"):
                value = value.detach()
            self._diagnostic_sums[key] = self._diagnostic_sums.get(key, 0.0) + value

    def log(self, logs, *args, **kwargs):
        import torch
        import torch.distributed as dist

        if self.diagnostic_every and "loss" in logs:
            now = time.monotonic()
            previous = getattr(self, "_last_step_log_time", None)
            if previous is not None:
                logs["runtime/step_wall_s"] = now - previous
            self._last_step_log_time = now
            step = self.state.global_step
            if step % self.diagnostic_every == 0 or step == self.state.max_steps:
                sums = getattr(self, "_diagnostic_sums", {})
                # Direct/confidence branch counts can differ by rank. Agree keys
                # before reduction and count candidates/tokens rather than ranks.
                key_lists = [None] * dist.get_world_size() if dist.is_initialized() else [sorted(sums)]
                if dist.is_initialized():
                    dist.all_gather_object(key_lists, sorted(sums))
                keys = sorted(set().union(*key_lists))
                device = next(self.model.parameters()).device
                packed = torch.stack([torch.as_tensor(sums.get(key, 0), device=device, dtype=torch.float64) for key in keys]).cpu()
                if dist.is_initialized():
                    if dist.get_backend() == "nccl":
                        packed = packed.to(device)
                    dist.all_reduce(packed)
                values = dict(zip(keys, packed.cpu().tolist()))
                summarized = summarize(values)
                for key, value in summarized.items():
                    logs["diag/" + key] = value
                objective = summarized.get("objective_lm", 0) + summarized.get("objective_confidence", 0)
                if objective > 0:
                    logs["runtime/trainer_logged_loss_to_mean_forward_loss"] = logs["loss"] / objective
                logs["runtime/gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
                started, start_step = getattr(self, "_diagnostic_window", (now, step))
                elapsed = now - started
                if elapsed > 0 and step > start_step:
                    logs["runtime/window_step_wall_s"] = elapsed / (step - start_step)
                    logs["runtime/input_tokens_per_s"] = values.get("input_tokens", 0) / elapsed
                    logs["runtime/micro_inputs_per_s"] = values.get("micro_samples", 0) / elapsed
                if torch.cuda.is_available():
                    memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], dtype=torch.float64)
                    if dist.is_initialized():
                        if dist.get_backend() == "nccl":
                            memory = memory.to(device)
                        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                    logs["runtime/max_rank_peak_allocated_gib"], logs["runtime/max_rank_peak_reserved_gib"] = (memory.cpu() / 1024**3).tolist()
                    torch.cuda.reset_peak_memory_stats()
                for i, group in enumerate(self.optimizer.param_groups):
                    logs[f"runtime/optimizer_group_{i}_lr"] = group["lr"]
                self._diagnostic_sums = {}
                self._diagnostic_window = (now, step)
        return super().log(logs, *args, **kwargs)
