#!/usr/bin/env python3
"""Evaluate a trained TimeLens2 arm with one video encode and one autoregressive decode.

For the joint arm, candidate confidence is read from final-normalized states
produced inside that same decode.  No teacher-forced replay is permitted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tvg_confidence.common import canonical_json, read_jsonl, sha256_file, stable_id, write_json  # noqa: E402
from tvg_confidence.instrumentation import _distribution_stats, _move_batch_to_model  # noqa: E402
from tvg_confidence.intervals import (  # noqa: E402
    CanonicalIntervalSet,
    IntervalParse,
    canonicalize_interval_set_to_duration,
    parse_interval_set_strict,
)
from tvg_confidence.semantic import map_token_semantics  # noqa: E402


DEFAULT_PROTOCOL = "timelens2_single_stage_generator_single_pass_eval_v1"
DEFAULT_INPUT_PROTOCOL = "timelens2_single_stage_generator_single_pass_eval_v1"
EXPECTED_MODEL_REVISION = "ddbb6cb944f13ce21e59e85da23c5f356107260e"
EXPECTED_VIDEO_CONTRACT = {
    "fps": 2.0,
    "max_frames": 120,
    "min_pixels": 4096,
    "max_pixels": 65536,
    "total_pixels": 7_864_320,
}
SPAN_ROLES = {
    "span_open",
    "start_value",
    "pair_delimiter",
    "end_value",
    "span_close",
}


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def interval_key(interval: Sequence[float]) -> tuple[float, float]:
    return round(float(interval[0]), 9), round(float(interval[1]), 9)


def complete_numeric_pairs(response: str) -> list[list[float]]:
    root_start = response.find("[")
    if root_start < 0:
        return []
    decoder = json.JSONDecoder()
    cursor = root_start + 1
    pairs: list[list[float]] = []
    while cursor < len(response):
        while cursor < len(response) and response[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(response) or response[cursor] == "]":
            break
        try:
            value, end = decoder.raw_decode(response, cursor)
        except json.JSONDecodeError:
            break
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
            or any(not math.isfinite(float(item)) for item in value)
        ):
            break
        pairs.append([float(value[0]), float(value[1])])
        cursor = int(end)
    return pairs


def strict_root_array_closed(response: str) -> bool:
    try:
        value = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(value, list)


def sampling_seed(base_seed: int, request_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{request_id}:0".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


class CandidateAwareJsonProcessor:
    """Force one outer close and EOS once online candidate stopping fires."""

    def __init__(
        self,
        tokenizer: Any,
        prompt_length: int,
        *,
        max_unique: int,
        duplicate_patience: int,
        max_total: int,
    ) -> None:
        import torch

        close_ids = tokenizer.encode("]", add_special_tokens=False)
        if len(close_ids) != 1:
            raise ValueError(f"outer close must be one token, got {close_ids}")
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id")
        self.torch = torch
        self.tokenizer = tokenizer
        self.last_length = int(prompt_length)
        self.close_token_id = int(close_ids[0])
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.max_unique = int(max_unique)
        self.duplicate_patience = int(duplicate_patience)
        self.max_total = int(max_total)
        self.text = ""
        self.observed_pairs = 0
        self.unique: set[tuple[float, float]] = set()
        self.duplicate_streak = 0
        self.phase = "observe"
        self.stop_reason: str | None = None
        self.forced_close = False

    def _force(self, scores: Any, token_id: int) -> Any:
        forced = self.torch.full_like(scores, -self.torch.inf)
        forced[:, int(token_id)] = 0.0
        return forced

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        current_length = int(input_ids.shape[1])
        new_text = ""
        if current_length > self.last_length:
            new_ids = input_ids[0, self.last_length : current_length].tolist()
            new_text = self.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self.text += new_text
            self.last_length = current_length

        if self.phase == "force_eos":
            self.phase = "done"
            return self._force(scores, self.eos_token_id)
        if self.phase == "done":
            return scores
        if "]" not in new_text or strict_root_array_closed(self.text):
            return scores

        pairs = complete_numeric_pairs(self.text)
        for interval in pairs[self.observed_pairs :]:
            key = interval_key(interval)
            if key in self.unique:
                self.duplicate_streak += 1
            else:
                self.unique.add(key)
                self.duplicate_streak = 0
        self.observed_pairs = len(pairs)

        reason = None
        if len(self.unique) >= self.max_unique:
            reason = "max_unique_candidates"
        elif len(pairs) >= self.max_total:
            reason = "max_total_candidates"
        elif self.duplicate_streak >= self.duplicate_patience:
            reason = "duplicate_patience"
        if reason is None:
            return scores

        self.stop_reason = reason
        self.forced_close = True
        self.phase = "force_eos"
        return self._force(scores, self.close_token_id)


class FixedHungarianConfidenceProcessor:
    """Constrain one decode to an exact fixed-K bbox/conf grammar.

    Literal syntax is forced, each coordinate digit remains a model decision,
    and every confidence position is restricted to one Yes or No token.  The
    original logits for allowed tokens are preserved so the normalized
    P(Yes)/(P(Yes)+P(No)) can be read from the same autoregressive decode.
    """

    def __init__(
        self, tokenizer: Any, prompt_length: int, *, candidate_count: int
    ) -> None:
        import torch

        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id")
        self.torch = torch
        self.prompt_length = int(prompt_length)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.candidate_count = int(candidate_count)
        if self.candidate_count not in {16, 32}:
            raise ValueError(f"unsupported fixed candidate count: {candidate_count}")

        def encoded(text: str) -> list[int]:
            ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
            decoded = tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not ids or decoded != text:
                raise ValueError(f"fixed-K literal is not exactly tokenizable: {text!r}")
            return ids

        digit_ids = []
        for digit in "0123456789":
            ids = encoded(digit)
            if len(ids) != 1:
                raise ValueError(f"fixed-K digit must be one token: {digit!r} -> {ids}")
            digit_ids.append(ids[0])
        yes_ids = encoded("Yes")
        no_ids = encoded("No")
        if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids[0] == no_ids[0]:
            raise ValueError(f"fixed-K Yes/No must be distinct single tokens: {yes_ids}/{no_ids}")
        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]

        self.steps: list[tuple[int, ...]] = []
        self.confidence_step_indices: list[int] = []

        def literal(text: str) -> None:
            self.steps.extend((token_id,) for token_id in encoded(text))

        def digits4() -> None:
            for _ in range(4):
                self.steps.append(tuple(digit_ids))

        literal('{"bbox":[')
        for slot in range(self.candidate_count):
            if slot:
                literal(",")
            literal('["')
            digits4()
            literal('","')
            digits4()
            literal('"]')
        literal('],"conf":[')
        for slot in range(self.candidate_count):
            if slot:
                literal(",")
            literal('"')
            self.confidence_step_indices.append(len(self.steps))
            self.steps.append((self.yes_token_id, self.no_token_id))
            literal('"')
        literal("]}")
        self.steps.append((self.eos_token_id,))

    @property
    def required_completion_tokens(self) -> int:
        return len(self.steps)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        step = int(input_ids.shape[1]) - self.prompt_length
        allowed = (
            self.steps[step]
            if 0 <= step < len(self.steps)
            else (self.eos_token_id,)
        )
        constrained = self.torch.full_like(scores, -self.torch.inf)
        for token_id in allowed:
            constrained[:, int(token_id)] = scores[:, int(token_id)]
        return constrained

    def confidence_rows(
        self, step_scores: Sequence[Any], completion_ids: Sequence[int]
    ) -> tuple[list[float], list[str]]:
        confidences: list[float] = []
        decisions: list[str] = []
        for step in self.confidence_step_indices:
            if step >= len(step_scores) or step >= len(completion_ids):
                raise RuntimeError("fixed-K decode ended before all confidence positions")
            logits = step_scores[step][0, [self.yes_token_id, self.no_token_id]].float()
            probability = self.torch.softmax(logits, dim=0)[0]
            confidences.append(float(probability.detach().cpu()))
            selected = int(completion_ids[step])
            if selected == self.yes_token_id:
                decisions.append("Yes")
            elif selected == self.no_token_id:
                decisions.append("No")
            else:
                raise RuntimeError("fixed-K confidence token escaped the grammar")
        return confidences, decisions


class Fixed16HungarianConfidenceProcessor(FixedHungarianConfidenceProcessor):
    """Backward-compatible fixed16 grammar used by the v6 evaluator."""

    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        super().__init__(tokenizer, prompt_length, candidate_count=16)


class Fixed32HungarianConfidenceProcessor(FixedHungarianConfidenceProcessor):
    """Exact fixed32 grammar used by the v7 evaluator."""

    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        super().__init__(tokenizer, prompt_length, candidate_count=32)


def metadata_value(metadata: Any, name: str) -> Any:
    if isinstance(metadata, dict):
        return metadata[name]
    return getattr(metadata, name)


def explicit_confidence_prompt(query: str) -> str:
    return (
        "You are given a video.\n"
        f'For the query "{query}", generate the plausible temporal candidates.\n'
        "Return only a JSON array. Every item must be an object with exactly two fields: "
        '"interval": [start, end] in seconds and "confidence": a number from 0 to 1. '
        "Confidence should rank how well that interval matches the query. Keep distinct true "
        "occurrences as distinct candidates; low-confidence alternatives may also be emitted.\n"
    )


def fixed16_hungarian_confidence_prompt(query: str) -> str:
    return (
        "You are given a video.\n"
        "Task: temporal retrieval.\n"
        f'Find distinct time spans in which the query "{query}" occurs.\n'
        "Return exactly 16 temporal candidates, then exactly 16 confidence decisions. "
        "Each boundary is a quoted four-digit normalized value from 0000 (video start) "
        "to 9999 (video end). Generate all bbox entries before making any confidence "
        "decision. Do not output any text before or after the object.\n"
        'Output format MUST be exactly: {"bbox":[["0000","0001"],...,['
        '"9998","9999"]],"conf":["Yes","No",...,"No"]}'
    )


def fixed32_hungarian_confidence_prompt(query: str) -> str:
    return (
        "You are given a video.\n"
        "Task: temporal retrieval.\n"
        f'Find distinct time spans in which the query "{query}" occurs.\n'
        "Return exactly 32 temporal candidates, then exactly 32 confidence decisions. "
        "Each boundary is a quoted four-digit normalized value from 0000 (video start) "
        "to 9999 (video end). Generate all bbox entries before making any confidence "
        "decision. Do not output any text before or after the object.\n"
        'Output format MUST be exactly: {"bbox":[["0000","0001"],...,['
        '"9998","9999"]],"conf":["Yes","No",...,"No"]}'
    )


def explicit_confidence_query(row: dict[str, Any]) -> str:
    query = row.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    prompt = str(row.get("prompt_text", ""))
    prefix = 'Find every distinct time span in which the query "'
    suffix = '" occurs.'
    start = prompt.find(prefix)
    end = prompt.find(suffix, start + len(prefix)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError("cannot recover query from frozen prompt_text")
    recovered = prompt[start + len(prefix) : end].strip()
    if not recovered:
        raise ValueError("recovered query is empty")
    return recovered


def messages(
    row: dict[str, Any],
    *,
    explicit_candidate_confidence: bool = False,
    fixed16_hungarian_confidence: bool = False,
    fixed32_hungarian_confidence: bool = False,
) -> list[dict[str, Any]]:
    if row["video_contract"] != EXPECTED_VIDEO_CONTRACT:
        raise ValueError("input video contract drifted")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(row["video_path"]),
                    **EXPECTED_VIDEO_CONTRACT,
                },
                {
                    "type": "text",
                    "text": (
                        fixed32_hungarian_confidence_prompt(
                            explicit_confidence_query(row)
                        )
                        if fixed32_hungarian_confidence
                        else fixed16_hungarian_confidence_prompt(
                            explicit_confidence_query(row)
                        )
                        if fixed16_hungarian_confidence
                        else explicit_confidence_prompt(explicit_confidence_query(row))
                        if explicit_candidate_confidence
                        else str(row["prompt_text"])
                    ),
                },
            ],
        }
    ]


def prepare(
    processor: Any,
    model: Any,
    row: dict[str, Any],
    *,
    explicit_candidate_confidence: bool = False,
    fixed16_hungarian_confidence: bool = False,
    fixed32_hungarian_confidence: bool = False,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from qwen_vl_utils import process_vision_info

    prompt_messages = messages(
        row,
        explicit_candidate_confidence=explicit_candidate_confidence,
        fixed16_hungarian_confidence=fixed16_hungarian_confidence,
        fixed32_hungarian_confidence=fixed32_hungarian_confidence,
    )
    rendered = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    patch_size = int(getattr(processor.image_processor, "patch_size", 16))
    images, videos, video_kwargs = process_vision_info(
        prompt_messages,
        image_patch_size=patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is None:
        raise ValueError("expected one decoded video")
    videos, metadata = zip(*videos)
    videos = list(videos)
    metadata = list(metadata)
    if len(videos) != 1 or len(metadata) != 1:
        raise ValueError("expected exactly one decoded video")
    inputs = processor(
        text=[rendered],
        images=images,
        videos=videos,
        video_metadata=metadata,
        do_resize=False,
        return_tensors="pt",
        **(video_kwargs or {}),
    )
    grid_tensor = inputs.get("video_grid_thw")
    if grid_tensor is None or list(grid_tensor.shape) != [1, 3]:
        raise ValueError("unexpected video grid")
    grid = [int(value) for value in grid_tensor[0].tolist()]
    indices = [int(value) for value in metadata_value(metadata[0], "frames_indices")]
    source_fps = float(metadata_value(metadata[0], "fps"))
    source_frames = int(metadata_value(metadata[0], "total_num_frames"))
    merge_size = int(
        getattr(
            getattr(processor, "video_processor", processor.image_processor),
            "merge_size",
            2,
        )
    )
    spatial_tokens = grid[1] * grid[2] // (merge_size * merge_size)
    expected_float = min(
        max(source_frames / source_fps * EXPECTED_VIDEO_CONTRACT["fps"], 4.0),
        float(EXPECTED_VIDEO_CONTRACT["max_frames"]),
        float(source_frames),
    )
    expected_frames = int(math.floor(expected_float / 2.0) * 2)
    contract = {
        "requested": dict(row["video_contract"]),
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "sampled_frames": len(indices),
        "sampled_source_frame_first": indices[0],
        "sampled_source_frame_last": indices[-1],
        "sampled_source_time_first_s": indices[0] / source_fps,
        "sampled_source_time_last_s": indices[-1] / source_fps,
        "source_fps": source_fps,
        "view_source_frame_count": source_frames,
        "expected_sampled_frames_from_smart_nframes": expected_frames,
        "video_grid_thw": [grid],
        "temporal_tokens": grid[0],
        "visual_tokens_per_temporal_patch": spatial_tokens,
        "merge_size": merge_size,
        "frame_indices_monotone": all(
            left <= right for left, right in zip(indices, indices[1:])
        ),
        "sampled_frames_exact_smart_nframes": len(indices) == expected_frames,
        "not_hard_capped_at_30": len(indices) != 30 if expected_frames != 30 else True,
    }
    valid = (
        2 <= len(indices) <= 120
        and 4 <= spatial_tokens <= 64
        and grid[0] == math.ceil(len(indices) / 2)
        and contract["frame_indices_monotone"]
        and contract["sampled_frames_exact_smart_nframes"]
        and contract["not_hard_capped_at_30"]
    )
    if not valid:
        raise RuntimeError(f"64-token/120-frame processor contract failed: {contract}")
    return _move_batch_to_model(inputs, model), rendered, contract


def make_head(checkpoint: Path, expected_sha256: str, device: Any) -> Any:
    import torch
    from safetensors.torch import load_file

    if sha256_file(checkpoint) != expected_sha256:
        raise SystemExit("confidence-head SHA mismatch")
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(2_560),
        torch.nn.Linear(2_560, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 4),
    )
    head.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    return head.to(device=device, dtype=torch.float32).eval()


def make_span_head(
    checkpoint: Path,
    expected_sha256: str,
    config_path: Path,
    expected_config_sha256: str,
    expected_adapter_model_sha256: str,
    expected_adapter_config_sha256: str,
    device: Any,
    *,
    full_model_manifest: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file

    if sha256_file(checkpoint) != expected_sha256:
        raise SystemExit("span-confidence-head SHA mismatch")
    if sha256_file(config_path) != expected_config_sha256:
        raise SystemExit("span-head-config SHA mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = config.get("protocol_version")
    if protocol == "timelens2_v13_grpo_online_span_head_v1":
        from head_checkpoint import validate_config
        checks = validate_config(config, expected_sha256, full_model_manifest)
        checks["exported_head_config"] = (
            (full_model_manifest or {}).get("online_head", {}).get("head_config_sha256")
            == expected_config_sha256
        )
        checks["no_adapter_transfer"] = not (
            expected_adapter_model_sha256 or expected_adapter_config_sha256
        )
    elif protocol == "timelens2_a2pp_joint_coverage_span_confidence_grpo_v8":
        checks = {
            "pooling": config.get("pooling")
            == "mean final-layer hidden over exact semantic interval tokens",
            "semantic_roles": set(config.get("semantic_roles") or []) == SPAN_ROLES,
            "architecture": config.get("architecture") == [2_560, 256, 1],
            "target": config.get("target") == "max candidate-to-GT tIoU >= 0.5",
            "loss": config.get("loss") == "binary_cross_entropy_with_logits",
            "hidden_gradient": config.get("hidden_gradient") == "detached",
        }
    elif protocol in {
        "timelens2_positive_only_variable_close_detached_confidence_sft_v1_1",
        "timelens2_bounded_coverage_pool_soft_confidence_sft_v1",
        "timelens2_bounded_coverage_pool_soft_confidence_sft_v1_2",
        "timelens2_bounded_coverage_pool_soft_confidence_sft_v1_3",
        "timelens2_bounded_pool_gt_binary_sft_v1",
    }:
        generator_identity = config.get("generator_identity") or {}
        checks = {
            "pooling": config.get("pooling")
            == "mean over target interval text tokens",
            "target": config.get("target") == (
                "max_gt_tiou_ge_0.5" if protocol == "timelens2_bounded_pool_gt_binary_sft_v1"
                else "verifier_v2_yes_probability"),
            "ranking_min_gap": float(config.get("ranking_min_gap", -1.0)) == 0.1,
            "hidden_gradient": config.get("generator_confidence_stop_gradient") is True,
            "architecture": config.get("head_architecture")
            == {
                "dropout": 0.1,
                "hidden_dim": 256,
                "input_dim": 2_560,
                "output_dim": 1,
            },
            "adapter_model": generator_identity.get("adapter_model_sha256")
            == expected_adapter_model_sha256,
            "adapter_config": generator_identity.get("adapter_config_sha256")
            == expected_adapter_config_sha256,
        }
        if protocol == "timelens2_bounded_pool_gt_binary_sft_v1":
            checks["pure_bce"] = float(config.get("ranking_weight", -1)) == 0.0
    else:
        generator_identity = config.get("generator_identity") or {}
        checks = {
            "pooling": config.get("pooling")
            == "mean over exact semantic interval tokens",
            "semantic_roles": set(config.get("semantic_roles") or []) == SPAN_ROLES,
            "hidden_layer": config.get("hidden_layer") == "final decoder layer",
            "hidden_size": int(config.get("hidden_size", -1)) == 2_560,
            "head_width": int(config.get("head_width", -1)) == 256,
            "target": config.get("target") == "target_correct_tiou05",
            "loss": config.get("loss") == "binary_cross_entropy_with_logits",
            "generator_frozen": config.get("generator_frozen") is True,
            "trainable_prefix": config.get("trainable_prefix") == "span_head.",
            "adapter_model": generator_identity.get("adapter_model_sha256")
            == expected_adapter_model_sha256,
            "adapter_config": generator_identity.get("adapter_config_sha256")
            == expected_adapter_config_sha256,
        }
    if not all(checks.values()):
        raise SystemExit(f"span-head config contract failed: {checks}")
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(2_560),
        torch.nn.Linear(2_560, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 1),
    )
    head.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    return head.to(device=device, dtype=torch.float32).eval(), config


def trim_completion(generated: Any, prompt_length: int) -> tuple[list[int], list[Any]]:
    completion = [
        int(value) for value in generated.sequences[0, prompt_length:].tolist()
    ]
    scores = list(generated.scores)
    if len(completion) < len(scores):
        raise RuntimeError("generated completion shorter than score sequence")
    return completion[: len(scores)], scores


def canonicalize_response_for_same_decode_scoring(
    response: str,
    duration_s: float,
    *,
    allow_truncated_interval_prefix: bool,
    allow_forced_close_trailing_separator: bool = False,
) -> tuple[CanonicalIntervalSet, str]:
    """Keep only complete generated pairs without inventing interval syntax.

    The synthetic root close is used solely to recover exact character/token
    semantics for already-complete pairs.  It never becomes a prediction and
    strict parse validity remains false for a truncated response.
    """

    strict = canonicalize_interval_set_to_duration(response, duration_s)
    forced_close_pairs = (
        complete_numeric_pairs(response)
        if allow_forced_close_trailing_separator
        else []
    )
    if strict.parse_valid and (strict.intervals or not forced_close_pairs):
        return strict, strict.normalization_status
    trailing_separator = bool(re.search(r",\s*\]\s*$", response))
    is_truncated_prefix = (
        allow_truncated_interval_prefix
        and strict.source_parse.parse_error_code == "truncated_output"
    )
    is_forced_close_boundary = bool(
        allow_forced_close_trailing_separator
        and forced_close_pairs
        and (
            trailing_separator
            or not strict.parse_valid
            or not strict.intervals
        )
    )
    if not is_truncated_prefix and not is_forced_close_boundary:
        return strict, "invalid"

    root_start = response.find("[")
    if root_start < 0:
        return strict, "invalid"
    decoder = json.JSONDecoder()
    cursor = root_start + 1
    complete_pairs: list[list[float]] = []
    last_pair_end: int | None = None
    while cursor < len(response):
        while cursor < len(response) and response[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(response) or response[cursor] == "]":
            break
        try:
            value, end = decoder.raw_decode(response, cursor)
        except json.JSONDecodeError:
            break
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
            or any(not math.isfinite(float(item)) for item in value)
        ):
            break
        complete_pairs.append([float(value[0]), float(value[1])])
        last_pair_end = int(end)
        cursor = int(end)
    if not complete_pairs or last_pair_end is None:
        return strict, "invalid"

    repaired_prefix = response[:last_pair_end] + "]"
    semantic_parse = parse_interval_set_strict(repaired_prefix)
    if not semantic_parse.syntax_valid or len(semantic_parse.intervals) != len(
        complete_pairs
    ):
        raise RuntimeError("truncated-prefix semantic reconstruction failed")
    normalization_status = (
        "forced_close_boundary_normalization"
        if is_forced_close_boundary
        else "truncated_prefix_salvage"
    )
    parse_error_code = (
        "forced_close_trailing_separator"
        if is_forced_close_boundary
        else "truncated_output"
    )
    source_parse = IntervalParse(
        raw_text=response,
        payload_text=semantic_parse.payload_text,
        payload_char_start=semantic_parse.payload_char_start,
        payload_char_end=semantic_parse.payload_char_end,
        intervals=complete_pairs,
        parse_valid=False,
        syntax_valid=True,
        parse_error_code=parse_error_code,
        parse_error_detail=(
            "retained exact complete pairs before the forced-close trailing separator"
            if is_forced_close_boundary
            else "retained only fully generated interval pairs before the incomplete suffix"
        ),
        char_regions=semantic_parse.char_regions,
        semantic_anchors=semantic_parse.semantic_anchors,
    )
    intervals: list[list[float]] = []
    source_span_indices: list[int] = []
    for source_span_index, (start, end) in enumerate(complete_pairs):
        if 0.0 <= start < end <= float(duration_s):
            intervals.append([start, end])
            source_span_indices.append(source_span_index)
    return (
        CanonicalIntervalSet(
            source_parse=source_parse,
            intervals=intervals,
            source_span_indices=source_span_indices,
            parse_valid=bool(intervals),
            normalization_status=(
                normalization_status if intervals else "invalid"
            ),
        ),
        normalization_status if intervals else "invalid",
    )


def same_decode_token_alignment_contract(
    status: str, canonical: CanonicalIntervalSet
) -> str | None:
    """Accept only token offsets proved exact for the states being scored.

    ``partial`` from ``align_token_offsets`` means every generated-token prefix
    decoded monotonically to the final response, but re-tokenizing the complete
    response chose a different suffix segmentation.  That proof is sufficient
    only for a forced-close normalization: all candidate endpoint/span anchors
    precede the synthetic root close, so the non-canonical suffix cannot change
    which generated token produced a cached state.  Ordinary partial/failed
    alignments remain rejected.
    """

    if status == "exact":
        return "retokenization_exact"
    if (
        status == "partial"
        and canonical.normalization_status == "forced_close_boundary_normalization"
    ):
        return "forced_close_monotone_generated_prefix_exact"
    return None


def parse_explicit_candidate_confidence(
    response: str,
    duration_s: float,
    *,
    allow_truncated_prefix: bool = False,
    allow_empty_after_duration_filter: bool = False,
) -> tuple[Any, list[dict[str, Any]], str]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        if not allow_truncated_prefix:
            raise ValueError(f"explicit confidence JSON parse failed: {exc}") from exc
        decoder = json.JSONDecoder()
        cursor = 1 if response.lstrip().startswith("[") else 0
        text = response.lstrip()
        payload = []
        while cursor < len(text):
            while cursor < len(text) and text[cursor] in " \t\r\n,":
                cursor += 1
            if cursor >= len(text) or text[cursor] == "]":
                break
            try:
                value, end = decoder.raw_decode(text, cursor)
            except json.JSONDecodeError:
                break
            payload.append(value)
            cursor = end
        if not payload:
            raise ValueError(
                f"explicit confidence JSON parse failed with no complete prefix candidate: {exc}"
            ) from exc
        parse_mode = "truncated_prefix_salvage"
    else:
        parse_mode = "strict"
    if not isinstance(payload, list) or not payload:
        raise ValueError("explicit confidence response must be a non-empty JSON array")
    intervals: list[list[float]] = []
    confidences: list[float] = []
    for index, candidate in enumerate(payload):
        if not isinstance(candidate, dict) or set(candidate) != {
            "interval",
            "confidence",
        }:
            raise ValueError(f"candidate {index} must contain exactly interval/confidence")
        interval = candidate["interval"]
        confidence = candidate["confidence"]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in interval
            )
        ):
            raise ValueError(f"candidate {index} interval is invalid")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"candidate {index} confidence is invalid")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"candidate {index} confidence is outside [0, 1]")
        intervals.append([float(interval[0]), float(interval[1])])
        confidences.append(confidence)
    canonical = canonicalize_interval_set_to_duration(
        json.dumps(intervals, separators=(",", ":")), duration_s
    )
    if not canonical.parse_valid or not canonical.intervals:
        if (
            allow_empty_after_duration_filter
            and canonical.source_parse.syntax_valid
            and canonical.source_parse.parse_error_code
            in {"out_of_range", "reversed_boundary"}
        ):
            return canonical, [], "duration_filter_empty"
        raise ValueError(
            "explicit confidence intervals are invalid: "
            f"{canonical.source_parse.parse_error_code}"
        )
    predictions = []
    for span_index, (source_span_index, interval) in enumerate(
        zip(canonical.source_span_indices, canonical.intervals)
    ):
        predictions.append(
            {
                "span_index": span_index,
                "source_span_index": int(source_span_index),
                "candidate_interval_source_s": [float(interval[0]), float(interval[1])],
                "generated_confidence": confidences[source_span_index],
                "confidence_source": "same_decode_explicit_json",
            }
        )
    return canonical, predictions, parse_mode


def parse_fixed_hungarian_confidence(
    response: str,
    duration_s: float,
    confidences: Sequence[float],
    decisions: Sequence[str],
    *,
    candidate_count: int,
) -> tuple[CanonicalIntervalSet, list[dict[str, Any]], dict[str, Any]]:
    """Validate exact fixed-K syntax and map normalized bins to seconds."""

    if candidate_count not in {16, 32}:
        raise ValueError(f"unsupported fixed candidate count: {candidate_count}")
    if len(confidences) != candidate_count or len(decisions) != candidate_count:
        raise ValueError(
            f"fixed{candidate_count} confidence vector must contain exactly "
            f"{candidate_count} entries"
        )
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fixed{candidate_count} JSON parse failed: {exc}") from exc
    if not isinstance(payload, dict) or list(payload) != ["bbox", "conf"]:
        raise ValueError(
            f"fixed{candidate_count} root must contain bbox then conf and no other keys"
        )
    bbox = payload["bbox"]
    labels = payload["conf"]
    if not isinstance(bbox, list) or len(bbox) != candidate_count:
        raise ValueError(
            f"fixed{candidate_count} bbox must contain exactly {candidate_count} candidates"
        )
    if not isinstance(labels, list) or len(labels) != candidate_count:
        raise ValueError(
            f"fixed{candidate_count} conf must contain exactly {candidate_count} decisions"
        )
    if labels != list(decisions) or any(value not in {"Yes", "No"} for value in labels):
        raise ValueError(
            f"fixed{candidate_count} decoded confidence labels do not match score positions"
        )
    canonical_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    if response != canonical_text:
        raise ValueError(
            f"fixed{candidate_count} response is not the exact canonical grammar"
        )

    source_intervals: list[list[float]] = []
    valid_intervals: list[list[float]] = []
    source_span_indices: list[int] = []
    invalid_indices: list[int] = []
    for index, pair in enumerate(bbox):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, str) or re.fullmatch(r"\d{4}", value) is None for value in pair)
        ):
            raise ValueError(
                f"fixed{candidate_count} bbox slot {index} is not two quoted four-digit values"
            )
        start_bin, end_bin = int(pair[0]), int(pair[1])
        start = duration_s * start_bin / 9_999.0
        end = duration_s * end_bin / 9_999.0
        source_intervals.append([start, end])
        if start_bin < end_bin:
            source_span_indices.append(index)
            valid_intervals.append([start, end])
        else:
            invalid_indices.append(index)

    source_parse = IntervalParse(
        raw_text=response,
        payload_text=response,
        payload_char_start=0,
        payload_char_end=len(response),
        intervals=source_intervals,
        parse_valid=not invalid_indices,
        syntax_valid=True,
        parse_error_code=("non_increasing_boundary" if invalid_indices else None),
        parse_error_detail=(
            f"invalid fixed{candidate_count} slots: {invalid_indices}"
            if invalid_indices
            else None
        ),
        char_regions=[],
        semantic_anchors=[],
    )
    canonical = CanonicalIntervalSet(
        source_parse=source_parse,
        intervals=valid_intervals,
        source_span_indices=source_span_indices,
        parse_valid=bool(valid_intervals),
        normalization_status=(
            "invalid_boundary_filter"
            if invalid_indices
            else f"exact_fixed{candidate_count}"
        ),
    )
    predictions = [
        {
            "span_index": span_index,
            "source_span_index": source_index,
            "fixed_candidate_slot_index": source_index,
            "fixed16_slot_index": source_index if candidate_count == 16 else None,
            "fixed32_slot_index": source_index if candidate_count == 32 else None,
            "candidate_interval_source_s": [float(interval[0]), float(interval[1])],
            "generated_confidence": float(confidences[source_index]),
            "generated_decision": str(decisions[source_index]),
            "confidence_source": "same_decode_yes_no_normalized_probability",
        }
        for span_index, (source_index, interval) in enumerate(
            zip(source_span_indices, valid_intervals)
        )
    ]
    diagnostics = {
        "fixed_candidate_count": candidate_count,
        "valid_candidate_count": len(valid_intervals),
        "invalid_boundary_count": len(invalid_indices),
        "invalid_boundary_indices": invalid_indices,
        "yes_decision_count": labels.count("Yes"),
        "no_decision_count": labels.count("No"),
        "grammar_exact": True,
    }
    return canonical, predictions, diagnostics


def parse_fixed16_hungarian_confidence(
    response: str,
    duration_s: float,
    confidences: Sequence[float],
    decisions: Sequence[str],
) -> tuple[CanonicalIntervalSet, list[dict[str, Any]], dict[str, Any]]:
    return parse_fixed_hungarian_confidence(
        response,
        duration_s,
        confidences,
        decisions,
        candidate_count=16,
    )


def parse_fixed32_hungarian_confidence(
    response: str,
    duration_s: float,
    confidences: Sequence[float],
    decisions: Sequence[str],
) -> tuple[CanonicalIntervalSet, list[dict[str, Any]], dict[str, Any]]:
    return parse_fixed_hungarian_confidence(
        response,
        duration_s,
        confidences,
        decisions,
        candidate_count=32,
    )


def score_endpoints_from_calls(
    *,
    tokenizer: Any,
    completion_ids: Sequence[int],
    response: str,
    duration_s: float,
    call_states: Sequence[Any],
    head: Any,
    device: Any,
    endpoint_anchor: str,
    canonical: CanonicalIntervalSet | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Map the configured semantic endpoint token to decode call j+1; never replay."""

    import torch

    canonical = canonical or canonicalize_interval_set_to_duration(
        response, duration_s
    )
    parsed = canonical.source_parse
    if not canonical.parse_valid or not canonical.intervals:
        return canonical, []
    _token_rows, anchors, status = map_token_semantics(
        tokenizer, completion_ids, response, parsed
    )
    endpoint_tokens = {
        int(anchor["span_index"]): int(anchor["prefix_token_end"])
        for anchor in anchors
        if anchor.get("semantic_event") == endpoint_anchor
        and anchor.get("span_index") is not None
        and isinstance(anchor.get("prefix_token_end"), int)
    }
    alignment_contract = same_decode_token_alignment_contract(status, canonical)
    if alignment_contract is None or sorted(endpoint_tokens) != list(
        range(len(parsed.intervals))
    ):
        raise RuntimeError(
            f"candidate {endpoint_anchor} mapping is not exact: "
            f"status={status}, normalization={canonical.normalization_status}, "
            f"mapped={sorted(endpoint_tokens)}, expected={list(range(len(parsed.intervals)))}"
        )
    rows: list[dict[str, Any]] = []
    endpoint_states = []
    for source_span_index in canonical.source_span_indices:
        token_index = endpoint_tokens[source_span_index]
        call_index = token_index + 1
        if call_index >= len(call_states):
            raise RuntimeError(
                "endpoint token was never consumed inside this decode; refusing replay"
            )
        state = call_states[call_index]
        if list(state.shape) != [1, 2_560]:
            raise RuntimeError(f"unexpected cached endpoint state shape: {list(state.shape)}")
        endpoint_states.append(state[0])
    tensor = torch.stack(endpoint_states).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = head(tensor).float().cpu()
    quality = torch.sigmoid(logits[:, 0]).tolist()
    ordinal = torch.sigmoid(logits[:, 1:]).tolist()
    for span_index, (source_span_index, interval, q, probabilities) in enumerate(
        zip(
            canonical.source_span_indices,
            canonical.intervals,
            quality,
            ordinal,
        )
    ):
        rows.append(
            {
                "span_index": span_index,
                "source_span_index": source_span_index,
                "candidate_interval_source_s": [float(interval[0]), float(interval[1])],
                "endpoint_anchor": endpoint_anchor,
                "completion_token_index_endpoint": endpoint_tokens[source_span_index],
                f"completion_token_index_{endpoint_anchor}": endpoint_tokens[
                    source_span_index
                ],
                "decode_call_index": endpoint_tokens[source_span_index] + 1,
                "token_alignment_status": status,
                "token_alignment_contract": alignment_contract,
                "predicted_quality": float(q),
                "probability_tiou_03": float(probabilities[0]),
                "probability_tiou_05": float(probabilities[1]),
                "probability_tiou_07": float(probabilities[2]),
                "state_source": "same_decode_final_norm",
            }
        )
    return canonical, rows


def score_span_hidden_from_calls(
    *,
    tokenizer: Any,
    completion_ids: Sequence[int],
    response: str,
    call_states: Sequence[Any],
    head: Any,
    device: Any,
    canonical: CanonicalIntervalSet,
    target_semantics: str = "max candidate-to-GT tIoU >= 0.5",
) -> list[dict[str, Any]]:
    """Pool exact interval-token states from this decode and apply the BCE head."""

    import torch

    if not canonical.parse_valid or not canonical.intervals:
        return []
    token_rows, _anchors, status = map_token_semantics(
        tokenizer, completion_ids, response, canonical.source_parse
    )
    alignment_contract = same_decode_token_alignment_contract(status, canonical)
    if alignment_contract is None:
        raise RuntimeError(
            "span token mapping is not exact: "
            f"status={status}, normalization={canonical.normalization_status}"
        )
    positions_by_span: dict[int, list[int]] = {
        int(index): [] for index in range(len(canonical.source_parse.intervals))
    }
    for token in token_rows:
        if token.span_index is None or token.semantic_role not in SPAN_ROLES:
            continue
        positions_by_span[int(token.span_index)].append(int(token.step_idx))

    pooled_states = []
    retained_positions: list[list[int]] = []
    for source_span_index in canonical.source_span_indices:
        positions = positions_by_span.get(int(source_span_index), [])
        if not positions:
            raise RuntimeError(
                f"empty exact token span for source span {source_span_index}"
            )
        states = []
        for token_index in positions:
            call_index = token_index + 1
            if call_index >= len(call_states):
                raise RuntimeError(
                    "span token was never consumed inside this decode; refusing replay"
                )
            state = call_states[call_index]
            if list(state.shape) != [1, 2_560]:
                raise RuntimeError(
                    f"unexpected cached span state shape: {list(state.shape)}"
                )
            states.append(state[0].float())
        pooled_states.append(torch.stack(states).mean(dim=0))
        retained_positions.append(positions)
    tensor = torch.stack(pooled_states).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        probabilities = torch.sigmoid(head(tensor).squeeze(-1)).float().cpu().tolist()
    return [
        {
            "span_index": span_index,
            "source_span_index": int(source_span_index),
            "candidate_interval_source_s": [
                float(interval[0]),
                float(interval[1]),
            ],
            "completion_token_indices_span": positions,
            "token_alignment_status": status,
            "token_alignment_contract": alignment_contract,
            "predicted_span_correctness_probability": float(probability),
            "state_source": "same_decode_final_layer_exact_span_hidden_mean",
            "target_semantics": target_semantics,
        }
        for span_index, (source_span_index, interval, positions, probability) in enumerate(
            zip(
                canonical.source_span_indices,
                canonical.intervals,
                retained_positions,
                probabilities,
            )
        )
    ]


def candidate_features_from_generation(
    *,
    tokenizer: Any,
    completion_ids: Sequence[int],
    response: str,
    duration_s: float,
    selected_logprobs: Sequence[float],
    canonical: CanonicalIntervalSet | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build the frozen no-extra-forward geometry/likelihood control features."""

    canonical = canonical or canonicalize_interval_set_to_duration(
        response, duration_s
    )
    parsed = canonical.source_parse
    if not canonical.parse_valid or not canonical.intervals:
        return canonical, []
    if len(selected_logprobs) != len(completion_ids):
        raise RuntimeError("generation logprob/token count mismatch")
    token_rows, anchors, status = map_token_semantics(
        tokenizer, completion_ids, response, parsed
    )
    after = {
        int(anchor["span_index"]): int(anchor["prefix_token_end"])
        for anchor in anchors
        if anchor.get("semantic_event") == "after_span"
        and anchor.get("span_index") is not None
        and isinstance(anchor.get("prefix_token_end"), int)
    }
    alignment_contract = same_decode_token_alignment_contract(status, canonical)
    if alignment_contract is None or sorted(after) != list(range(len(parsed.intervals))):
        raise RuntimeError(
            "candidate feature endpoint mapping is not exact: "
            f"status={status}, normalization={canonical.normalization_status}, "
            f"mapped={sorted(after)}, expected={list(range(len(parsed.intervals)))}"
        )
    per_span_tokens: dict[int, list[int]] = {
        span_index: [] for span_index in range(len(parsed.intervals))
    }
    for token_row in token_rows:
        if token_row.span_index is not None:
            per_span_tokens[int(token_row.span_index)].append(int(token_row.step_idx))
    if any(not indices for indices in per_span_tokens.values()):
        raise RuntimeError("candidate feature mapping has an empty token span")
    response_mean = statistics.fmean(float(value) for value in selected_logprobs)
    source_count = len(parsed.intervals)
    features = []
    for span_index, (source_span_index, interval) in enumerate(
        zip(canonical.source_span_indices, canonical.intervals)
    ):
        indices = per_span_tokens[source_span_index]
        start, end = map(float, interval)
        features.append(
            {
                "span_index": span_index,
                "source_span_index": source_span_index,
                "candidate_interval_source_s": [start, end],
                "completion_token_index_after_span": after[source_span_index],
                "feature_anchor": "after_span",
                "token_alignment_status": status,
                "token_alignment_contract": alignment_contract,
                "candidate_position_normalized": (source_span_index + 1) / source_count,
                "candidate_start_normalized": start / duration_s,
                "candidate_end_normalized": end / duration_s,
                "candidate_length_normalized": (end - start) / duration_s,
                "candidate_generation_mean_logprob": statistics.fmean(
                    float(selected_logprobs[index]) for index in indices
                ),
                "candidate_end_token_logprob": float(
                    selected_logprobs[after[source_span_index]]
                ),
                "candidate_after_span_token_logprob": float(
                    selected_logprobs[after[source_span_index]]
                ),
                "response_generation_mean_logprob": response_mean,
                "generated_candidate_count": source_count,
                "feature_source": "same_decode_no_extra_model_call",
            }
        )
    return canonical, features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-version", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--expected-input-protocol-version", default=DEFAULT_INPUT_PROTOCOL
    )
    parser.add_argument(
        "--endpoint-anchor", choices=("after_span", "after_end"), default="after_span"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--arm", choices=("direct", "joint"), required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--expected-adapter-model-sha256")
    parser.add_argument("--expected-adapter-config-sha256")
    parser.add_argument("--full-model-manifest", type=Path)
    parser.add_argument("--expected-full-model-manifest-sha256")
    parser.add_argument("--confidence-head", type=Path)
    parser.add_argument("--expected-confidence-head-sha256")
    parser.add_argument("--span-confidence-head", type=Path)
    parser.add_argument("--expected-span-confidence-head-sha256")
    parser.add_argument("--span-head-config", type=Path)
    parser.add_argument("--expected-span-head-config-sha256")
    parser.add_argument("--span-head-source-adapter-model-sha256")
    parser.add_argument("--span-head-source-adapter-config-sha256")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--smoke-max-rows", type=int)
    parser.add_argument("--paper-main-eligible", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing-output", action="store_true")
    parser.add_argument("--explicit-candidate-confidence", action="store_true")
    parser.add_argument("--fixed16-hungarian-confidence", action="store_true")
    parser.add_argument("--fixed32-hungarian-confidence", action="store_true")
    parser.add_argument(
        "--salvage-truncated-explicit-confidence", action="store_true"
    )
    parser.add_argument(
        "--salvage-empty-duration-filter-explicit-confidence", action="store_true"
    )
    parser.add_argument(
        "--salvage-truncated-interval-prefix", action="store_true"
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sampling-seed", type=int, default=20260826)
    parser.add_argument("--candidate-aware-stop", action="store_true")
    parser.add_argument("--candidate-max-unique", type=int, default=32)
    parser.add_argument("--candidate-duplicate-patience", type=int, default=3)
    parser.add_argument("--candidate-max-total", type=int, default=40)
    parser.add_argument("--exact-dedup", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    args = parser.parse_args()
    fixed_hungarian_confidence = bool(
        args.fixed16_hungarian_confidence or args.fixed32_hungarian_confidence
    )
    fixed_candidate_count = (
        16
        if args.fixed16_hungarian_confidence
        else 32
        if args.fixed32_hungarian_confidence
        else None
    )
    if args.fixed16_hungarian_confidence and args.fixed32_hungarian_confidence:
        raise SystemExit("fixed16 and fixed32 modes are exclusive")
    if args.model_revision != EXPECTED_MODEL_REVISION:
        raise SystemExit("model revision drifted")
    if sha256_file(args.input) != args.expected_input_sha256:
        raise SystemExit("input SHA mismatch")
    full_model = None
    if args.full_model_manifest is not None:
        if any((args.adapter_dir, args.expected_adapter_model_sha256,
                args.expected_adapter_config_sha256)):
            raise SystemExit("full-model inference cannot also load an adapter")
        if sha256_file(args.full_model_manifest) != args.expected_full_model_manifest_sha256:
            raise SystemExit("full-model manifest SHA mismatch")
        full_model = json.loads(args.full_model_manifest.read_text())
        if full_model["status"] != "complete" or Path(full_model["model_path"]).resolve() != Path(args.model_path).resolve():
            raise SystemExit("full-model export is incomplete or belongs to another path")
        for name, digest in full_model["model_files"].items():
            if sha256_file(Path(args.model_path) / name) != digest:
                raise SystemExit("full-model file SHA mismatch: " + name)
    else:
        if args.expected_full_model_manifest_sha256 or not all((args.adapter_dir,
                args.expected_adapter_model_sha256, args.expected_adapter_config_sha256)):
            raise SystemExit("adapter inference requires its directory and both SHAs")
        adapter_model = args.adapter_dir / "adapter_model.safetensors"
        adapter_config = args.adapter_dir / "adapter_config.json"
        if (sha256_file(adapter_model) != args.expected_adapter_model_sha256
                or sha256_file(adapter_config) != args.expected_adapter_config_sha256):
            raise SystemExit("adapter artifact SHA mismatch")
    rows = list(read_jsonl(args.input))
    if len(rows) != args.expected_rows or len(
        {str(row["request_id"]) for row in rows}
    ) != len(rows):
        raise SystemExit("input row/identity mismatch")
    if any(
        row.get("protocol_version") != args.expected_input_protocol_version
        for row in rows
    ):
        raise SystemExit("input protocol mismatch")
    if args.output_dir.exists() and not args.resume_existing_output:
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    if args.resume_existing_output and not args.output_dir.exists():
        raise SystemExit("resume requested but output directory does not exist")
    if args.arm == "joint" and args.confidence_head is None and args.span_confidence_head is None:
        raise SystemExit("joint arm requires an endpoint or span confidence head")
    if (args.confidence_head is None) != (args.expected_confidence_head_sha256 is None):
        raise SystemExit("endpoint confidence head requires checkpoint and SHA together")
    if args.arm == "direct" and args.confidence_head is not None:
        raise SystemExit("direct arm must not load a confidence head")
    if args.arm == "direct" and args.expected_confidence_head_sha256 is not None:
        raise SystemExit("direct arm must not declare a confidence-head SHA")
    if args.explicit_candidate_confidence and args.arm != "direct":
        raise SystemExit("explicit candidate-confidence generation is a direct LM arm")
    if fixed_hungarian_confidence and args.arm != "direct":
        raise SystemExit("fixed-K Hungarian confidence generation is a direct LM arm")
    if fixed_hungarian_confidence and args.explicit_candidate_confidence:
        raise SystemExit("fixed-K and explicit candidate-confidence modes are exclusive")
    if fixed_hungarian_confidence and any(
        value is not None
        for value in (args.confidence_head, args.span_confidence_head)
    ):
        raise SystemExit("fixed-K confidence is read from LM Yes/No logits, not an external head")
    span_arguments = (
        args.span_confidence_head,
        args.expected_span_confidence_head_sha256,
        args.span_head_config,
        args.expected_span_head_config_sha256,
    )
    if any(value is not None for value in span_arguments) and not all(
        value is not None for value in span_arguments
    ):
        raise SystemExit("span-confidence head requires checkpoint/config and both SHAs")
    span_source_arguments = (
        args.span_head_source_adapter_model_sha256,
        args.span_head_source_adapter_config_sha256,
    )
    if any(value is not None for value in span_source_arguments) and not all(
        value is not None for value in span_source_arguments
    ):
        raise SystemExit("span-head source adapter identity requires both SHAs")
    if args.span_confidence_head is not None and args.arm != "joint":
        raise SystemExit("span-confidence head is supported only for the joint arm")
    if args.span_confidence_head is not None and args.explicit_candidate_confidence:
        raise SystemExit("span-confidence head cannot score explicit-confidence JSON")
    if args.salvage_truncated_interval_prefix and args.explicit_candidate_confidence:
        raise SystemExit("interval-prefix salvage is for interval-only generation")
    if fixed_hungarian_confidence and (
        args.salvage_truncated_interval_prefix
        or args.salvage_truncated_explicit_confidence
        or args.salvage_empty_duration_filter_explicit_confidence
        or args.candidate_aware_stop
    ):
        if args.fixed32_hungarian_confidence:
            raise SystemExit("fixed32 constrained decoding forbids salvage and candidate stopping")
        raise SystemExit("fixed16 constrained decoding forbids salvage and candidate stopping")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise SystemExit("max-new-tokens must be positive")
    if args.do_sample and not (
        0.0 < args.temperature <= 2.0 and 0.0 < args.top_p <= 1.0
    ):
        raise SystemExit("invalid sampling parameters")
    if args.candidate_aware_stop:
        if args.explicit_candidate_confidence:
            raise SystemExit("candidate-aware stop requires interval-only JSON")
        if args.max_new_tokens != 512:
            raise SystemExit("candidate-aware stop freezes max-new-tokens=512")
        if min(
            args.candidate_max_unique,
            args.candidate_duplicate_patience,
            args.candidate_max_total,
        ) <= 0:
            raise SystemExit("candidate stopping parameters must be positive")
        if args.candidate_max_unique > args.candidate_max_total:
            raise SystemExit("max unique candidates cannot exceed max total candidates")
    source_rows = len(rows)
    if args.smoke_max_rows is not None:
        if not 1 <= args.smoke_max_rows < source_rows:
            raise SystemExit("smoke-max-rows must be positive and smaller than the frozen input")
        rows = rows[: args.smoke_max_rows]

    os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
    import torch
    import transformers
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        LogitsProcessorList,
    )

    if transformers.__version__ != "4.57.1":
        raise SystemExit(f"runtime requires transformers 4.57.1, got {transformers.__version__}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=args.attention_backend,
    )
    # Newer PEFT versions serialize additive fields such as
    # ``corda_config=None``.  The A100 inference image can be older.  After the
    # frozen source hashes have been verified above, create a runtime-only view
    # that removes fields this installed LoraConfig cannot consume.  Refuse to
    # drop any non-default value, so checkpoint semantics cannot silently drift.
    if full_model is not None:
        model = base.eval()
        model.requires_grad_(False)
        backbone = model
    else:
        adapter_load_dir = args.adapter_dir
        adapter_config_payload = json.loads(adapter_config.read_text())
        supported_lora_fields = set(inspect.signature(LoraConfig).parameters)
        unsupported_lora_fields = sorted(set(adapter_config_payload) - supported_lora_fields)
        unsupported_nondefault = {
            key: adapter_config_payload[key]
            for key in unsupported_lora_fields
            if adapter_config_payload[key] not in (None, False, [], {})
        }
        if unsupported_nondefault:
            raise SystemExit("installed PEFT cannot preserve non-default adapter config fields: "
                             + json.dumps(unsupported_nondefault, sort_keys=True))
        with tempfile.TemporaryDirectory(prefix="tvg-peft-adapter-") as temporary_adapter:
            if unsupported_lora_fields:
                temporary_adapter_path = Path(temporary_adapter)
                filtered_config = {key: value for key, value in adapter_config_payload.items()
                                   if key not in unsupported_lora_fields}
                (temporary_adapter_path / "adapter_config.json").write_text(
                    json.dumps(filtered_config, indent=2, sort_keys=True) + "\n")
                os.symlink(adapter_model, temporary_adapter_path / adapter_model.name)
                adapter_load_dir = temporary_adapter_path
            model = PeftModel.from_pretrained(base, str(adapter_load_dir), is_trainable=False).eval()
        backbone = model.get_base_model()
    device = next(model.parameters()).device
    text_config = getattr(backbone.config, "text_config", backbone.config)
    decoder_hidden_size = int(text_config.hidden_size)
    head = (
        make_head(
            args.confidence_head,
            args.expected_confidence_head_sha256,
            device,
        )
        if args.confidence_head is not None
        else None
    )
    span_head_source_adapter_model_sha256 = (
        args.span_head_source_adapter_model_sha256
        or args.expected_adapter_model_sha256
    )
    span_head_source_adapter_config_sha256 = (
        args.span_head_source_adapter_config_sha256
        or args.expected_adapter_config_sha256
    )
    span_head, span_head_config = (
        make_span_head(
            args.span_confidence_head,
            args.expected_span_confidence_head_sha256,
            args.span_head_config,
            args.expected_span_head_config_sha256,
            span_head_source_adapter_model_sha256,
            span_head_source_adapter_config_sha256,
            device,
            full_model_manifest=full_model,
        )
        if args.span_confidence_head is not None
        else (None, None)
    )
    identity = {
        "protocol_version": args.protocol_version,
        "input_protocol_version": args.expected_input_protocol_version,
        "endpoint_anchor": args.endpoint_anchor,
        "cheap_control_feature_anchor": "after_span",
        "arm": args.arm,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "full_model_manifest_sha256": args.expected_full_model_manifest_sha256,
        "full_model_checkpoint_marker_sha256": full_model["checkpoint_marker_sha256"] if full_model else None,
        "adapter_loaded": full_model is None,
        "adapter_model_sha256": args.expected_adapter_model_sha256,
        "adapter_config_sha256": args.expected_adapter_config_sha256,
        "confidence_head_sha256": args.expected_confidence_head_sha256,
        "span_confidence_head_sha256": args.expected_span_confidence_head_sha256,
        "span_head_config_sha256": args.expected_span_head_config_sha256,
        "span_head_training_protocol_version": (
            span_head_config.get("protocol_version") if span_head_config else None
        ),
        "span_head_source_adapter_model_sha256": (
            span_head_source_adapter_model_sha256 if span_head is not None else None
        ),
        "span_head_source_adapter_config_sha256": (
            span_head_source_adapter_config_sha256 if span_head is not None else None
        ),
        "span_head_cross_generator_transfer": bool(
            span_head is not None
            and (
                span_head_source_adapter_model_sha256
                != args.expected_adapter_model_sha256
                or span_head_source_adapter_config_sha256
                != args.expected_adapter_config_sha256
            )
        ),
        "input_sha256": args.expected_input_sha256,
        "input_rows_frozen": source_rows,
        "input_rows_executed": len(rows),
        "smoke_only": args.smoke_max_rows is not None,
        "paper_main_eligible": args.paper_main_eligible,
        "explicit_candidate_confidence": args.explicit_candidate_confidence,
        "fixed16_hungarian_confidence": args.fixed16_hungarian_confidence,
        "fixed32_hungarian_confidence": args.fixed32_hungarian_confidence,
        "fixed_candidate_count": fixed_candidate_count,
        "salvage_truncated_explicit_confidence": args.salvage_truncated_explicit_confidence,
        "salvage_empty_duration_filter_explicit_confidence": args.salvage_empty_duration_filter_explicit_confidence,
        "salvage_truncated_interval_prefix": args.salvage_truncated_interval_prefix,
        "max_new_tokens_override": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "sampling_seed": args.sampling_seed if args.do_sample else None,
        "candidate_aware_stop": args.candidate_aware_stop,
        "candidate_max_unique": (
            args.candidate_max_unique if args.candidate_aware_stop else None
        ),
        "candidate_duplicate_patience": (
            args.candidate_duplicate_patience if args.candidate_aware_stop else None
        ),
        "candidate_max_total": (
            args.candidate_max_total if args.candidate_aware_stop else None
        ),
        "exact_dedup": args.exact_dedup,
        "forced_close_boundary_policy": (
            "retain exact complete pairs; separately mark trailing-separator normalization"
            if args.candidate_aware_stop
            else None
        ),
        "attention_backend": args.attention_backend,
        "runner_sha256": sha256_file(Path(__file__)),
        "video_contract": EXPECTED_VIDEO_CONTRACT,
        "inference_contract": {
            "video_encodes": 1,
            "autoregressive_decodes": 1,
            "teacher_forced_replays": 0,
            "confidence_heads_scored": (
                ["lm_yes_no_normalized_probability"]
                if fixed_hungarian_confidence
                else ["endpoint_tiou_ordinal", "span_hidden_binary_tiou05"]
                if span_head is not None
                else ["endpoint_tiou_ordinal"] if head is not None else []
            ),
        },
    }
    output_path = args.output_dir / "outputs.jsonl"
    resume_rows: list[dict[str, Any]] = []
    if args.resume_existing_output:
        old_manifest = json.loads(
            (args.output_dir / "run-manifest.json").read_text(encoding="utf-8")
        )
        old_identity = old_manifest.get("config_identity", {})
        contract_keys = (
            "protocol_version",
            "input_protocol_version",
            "endpoint_anchor",
            "arm",
            "model_path",
            "model_revision",
            "adapter_model_sha256",
            "adapter_config_sha256",
            "confidence_head_sha256",
            "span_confidence_head_sha256",
            "span_head_config_sha256",
            "span_head_training_protocol_version",
            "span_head_source_adapter_model_sha256",
            "span_head_source_adapter_config_sha256",
            "span_head_cross_generator_transfer",
            "input_sha256",
            "input_rows_frozen",
            "input_rows_executed",
            "smoke_only",
            "paper_main_eligible",
            "explicit_candidate_confidence",
            "fixed16_hungarian_confidence",
            "fixed32_hungarian_confidence",
            "fixed_candidate_count",
            "salvage_truncated_explicit_confidence",
            "salvage_empty_duration_filter_explicit_confidence",
            "salvage_truncated_interval_prefix",
            "max_new_tokens_override",
            "do_sample",
            "temperature",
            "top_p",
            "sampling_seed",
            "candidate_aware_stop",
            "candidate_max_unique",
            "candidate_duplicate_patience",
            "candidate_max_total",
            "exact_dedup",
            "forced_close_boundary_policy",
            "attention_backend",
            "video_contract",
            "inference_contract",
        )
        if full_model is not None:
            contract_keys += ("full_model_manifest_sha256", "full_model_checkpoint_marker_sha256", "adapter_loaded")
        if any(old_identity.get(key) != identity.get(key) for key in contract_keys):
            raise SystemExit("resume config identity mismatch")
        resume_rows = list(read_jsonl(output_path))
        if not resume_rows or len(resume_rows) >= len(rows):
            raise SystemExit("resume output length is not a strict non-empty prefix")
        for position, (source, completed) in enumerate(zip(rows, resume_rows)):
            if (
                completed.get("request_id") != source.get("request_id")
                or completed.get("execution_position") != position
                or completed.get("infer_status") != "success"
                or completed.get("run_id") != old_manifest.get("run_id")
            ):
                raise SystemExit(f"resume prefix mismatch at position {position}")
        identity = old_identity
        run_id = str(old_manifest["run_id"])
        manifest = old_manifest
        manifest["status"] = "running"
        manifest["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["resumed_from_rows"] = len(resume_rows)
    else:
        args.output_dir.mkdir(parents=True)
        run_id = stable_id("timelens2_single_stage_generator_single_pass_eval", identity)
        manifest = {
            "schema_version": 1,
            "protocol_version": args.protocol_version,
            "run_id": run_id,
            "status": "running",
            "config_identity": identity,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.cuda.get_device_name(0),
            },
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "paper_main_eligible": args.paper_main_eligible,
        }
    write_json(args.output_dir / "run-manifest.json", manifest)
    success = len(resume_rows)
    failed = 0
    canonical_valid = sum(bool(row.get("parse_valid")) for row in resume_rows)
    strict_valid = sum(bool(row.get("strict_parse_valid")) for row in resume_rows)
    duration_filtered = sum(
        bool(row.get("normalization_applied")) for row in resume_rows
    )
    for position, row in enumerate(rows[len(resume_rows) :], start=len(resume_rows)):
        # Drop every per-row CUDA reference before decoding the next video.  This
        # matters for VUE: one 1-FPS prompt can exceed 120k tokens and the next
        # video's preprocessing otherwise overlaps the previous generation's
        # score tensors/call-state cache.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        pre_row_memory_allocated = int(torch.cuda.memory_allocated(device))
        pre_row_memory_reserved = int(torch.cuda.memory_reserved(device))
        prepared = rendered = processor_contract = generated = None
        candidate_stopper = fixed_processor = call_states = completion_ids = step_scores = None
        response = parsed = endpoint_predictions = span_predictions = None
        interval_parse_mode = explicit_confidence_parse_mode = None
        fixed_diagnostics = None
        decode_stats = selected_logprobs = feature_parse = candidate_features = None
        raw_candidate_intervals = raw_endpoint_predictions = raw_span_predictions = None
        result = None
        try:
            end_to_end_started = time.perf_counter()
            preprocessing_started = end_to_end_started
            prepared, rendered, processor_contract = prepare(
                processor,
                model,
                row,
                explicit_candidate_confidence=args.explicit_candidate_confidence,
                fixed16_hungarian_confidence=args.fixed16_hungarian_confidence,
                fixed32_hungarian_confidence=args.fixed32_hungarian_confidence,
            )
            preprocessing_elapsed = time.perf_counter() - preprocessing_started
            # Inline generation here so the call-state tensors remain available
            # until endpoint scoring is complete.
            prompt_length = int(prepared["input_ids"].shape[1])
            if row.get("context_token_budget") is not None:
                completion_budget = args.max_new_tokens or int(row["decode"]["max_new_tokens"])
                if prompt_length + completion_budget > int(row["context_token_budget"]):
                    raise RuntimeError("prompt plus completion budget exceeds the frozen context limit")
            row_sampling_seed = (
                sampling_seed(args.sampling_seed, str(row["request_id"]))
                if args.do_sample
                else None
            )
            if row_sampling_seed is not None:
                torch.manual_seed(row_sampling_seed)
                torch.cuda.manual_seed_all(row_sampling_seed)
            candidate_stopper = (
                CandidateAwareJsonProcessor(
                    processor.tokenizer,
                    prompt_length,
                    max_unique=args.candidate_max_unique,
                    duplicate_patience=args.candidate_duplicate_patience,
                    max_total=args.candidate_max_total,
                )
                if args.candidate_aware_stop
                else None
            )
            fixed_processor = (
                FixedHungarianConfidenceProcessor(
                    processor.tokenizer,
                    prompt_length,
                    candidate_count=int(fixed_candidate_count),
                )
                if fixed_hungarian_confidence
                else None
            )
            if (
                fixed_processor is not None
                and args.max_new_tokens is not None
                and args.max_new_tokens < fixed_processor.required_completion_tokens
            ):
                raise RuntimeError(
                    "max-new-tokens is smaller than the fixed-K constrained grammar: "
                    f"{args.max_new_tokens} < {fixed_processor.required_completion_tokens}"
                )
            call_states = []
            vision_encode_calls = 0

            def state_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
                hidden = output.last_hidden_state
                if not call_states:
                    # The prefill state is used only for prompt-length alignment;
                    # no candidate confidence reads it.  Retaining the full
                    # [prompt_tokens, 2560] tensor costs hundreds of MiB on long
                    # VUE videos, so keep a shape-only sentinel instead.
                    call_states.append(
                        torch.empty(
                            (int(hidden.shape[-2]), 0),
                            dtype=torch.uint8,
                            device="cpu",
                        )
                    )
                else:
                    call_states.append(
                        hidden[0].detach().to("cpu").contiguous()
                    )

            def vision_hook(_module: Any, _args: tuple[Any, ...], _output: Any) -> None:
                nonlocal vision_encode_calls
                vision_encode_calls += 1

            state_handle = backbone.model.register_forward_hook(state_hook)
            vision_handle = backbone.model.visual.register_forward_hook(vision_hook)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": (
                        args.max_new_tokens
                        if args.max_new_tokens is not None
                        else int(row["decode"]["max_new_tokens"])
                    ),
                    "do_sample": args.do_sample,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                }
                if args.do_sample:
                    generation_kwargs.update(
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                if candidate_stopper is not None:
                    generation_kwargs["logits_processor"] = LogitsProcessorList(
                        [candidate_stopper]
                    )
                elif fixed_processor is not None:
                    generation_kwargs["logits_processor"] = LogitsProcessorList(
                        [fixed_processor]
                    )
                with torch.inference_mode():
                    generated = model.generate(
                        **prepared,
                        **generation_kwargs,
                    )
                torch.cuda.synchronize(device)
            finally:
                state_handle.remove()
                vision_handle.remove()
            elapsed = time.perf_counter() - started
            completion_ids, step_scores = trim_completion(generated, prompt_length)
            if len(call_states) != len(step_scores):
                raise RuntimeError("same-decode call/state alignment failed")
            if not call_states or int(call_states[0].shape[0]) != prompt_length:
                raise RuntimeError("same-decode prefill state alignment failed")
            if any(list(state.shape) != [1, decoder_hidden_size] for state in call_states[1:]):
                raise RuntimeError("same-decode cached state shape failed")
            if vision_encode_calls != 1:
                raise RuntimeError(
                    f"single-video-encode contract failed: {vision_encode_calls} != 1"
                )
            response = processor.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if fixed_hungarian_confidence:
                fixed_confidences, fixed_decisions = fixed_processor.confidence_rows(
                    step_scores,
                    completion_ids,
                )
                parsed, endpoint_predictions, fixed_diagnostics = parse_fixed_hungarian_confidence(
                    response,
                    float(row["source_duration_s"]),
                    fixed_confidences,
                    fixed_decisions,
                    candidate_count=int(fixed_candidate_count),
                )
                explicit_confidence_parse_mode = None
                interval_parse_mode = f"exact_fixed{fixed_candidate_count}_constrained"
                span_predictions = []
            elif args.explicit_candidate_confidence:
                parsed, endpoint_predictions, explicit_confidence_parse_mode = parse_explicit_candidate_confidence(
                    response,
                    float(row["source_duration_s"]),
                    allow_truncated_prefix=args.salvage_truncated_explicit_confidence,
                    allow_empty_after_duration_filter=args.salvage_empty_duration_filter_explicit_confidence,
                )
                interval_parse_mode = explicit_confidence_parse_mode
                span_predictions: list[dict[str, Any]] = []
            else:
                explicit_confidence_parse_mode = None
                parsed, interval_parse_mode = canonicalize_response_for_same_decode_scoring(
                    response,
                    float(row["source_duration_s"]),
                    allow_truncated_interval_prefix=args.salvage_truncated_interval_prefix,
                    allow_forced_close_trailing_separator=bool(
                        candidate_stopper is not None
                        and candidate_stopper.forced_close
                    ),
                )
                if head is not None:
                    _endpoint_parse, endpoint_predictions = score_endpoints_from_calls(
                        tokenizer=processor.tokenizer,
                        completion_ids=completion_ids,
                        response=response,
                        duration_s=float(row["source_duration_s"]),
                        call_states=call_states,
                        head=head,
                        device=device,
                        endpoint_anchor=args.endpoint_anchor,
                        canonical=parsed,
                    )
                    if (
                        _endpoint_parse.parse_valid != parsed.parse_valid
                        or _endpoint_parse.intervals != parsed.intervals
                    ):
                        raise RuntimeError("endpoint confidence parse alignment failed")
                else:
                    endpoint_predictions = []
                span_predictions = (
                    score_span_hidden_from_calls(
                        tokenizer=processor.tokenizer,
                        completion_ids=completion_ids,
                        response=response,
                        call_states=call_states,
                        head=span_head,
                        device=device,
                        canonical=parsed,
                        target_semantics=str(span_head_config["target"]),
                    )
                    if span_head is not None
                    else []
                )
            decode_stats = [
                _distribution_stats(scores[0], token_id)
                for scores, token_id in zip(step_scores, completion_ids)
            ]
            selected_logprobs = [
                float(item["selected_logprob"]) for item in decode_stats
            ]
            if fixed_hungarian_confidence:
                feature_parse, candidate_features = parsed, []
            elif args.explicit_candidate_confidence:
                feature_parse, candidate_features = parsed, []
            else:
                feature_parse, candidate_features = candidate_features_from_generation(
                    tokenizer=processor.tokenizer,
                    completion_ids=completion_ids,
                    response=response,
                    duration_s=float(row["source_duration_s"]),
                    selected_logprobs=selected_logprobs,
                    canonical=parsed,
                )
            if (
                feature_parse.parse_valid != parsed.parse_valid
                or feature_parse.intervals != parsed.intervals
            ):
                raise RuntimeError("confidence/cheap-control parse mismatch")
            if (
                endpoint_predictions
                and not args.explicit_candidate_confidence
                and not fixed_hungarian_confidence
                and len(endpoint_predictions) != len(candidate_features)
            ):
                raise RuntimeError("confidence/cheap-control candidate alignment mismatch")
            if span_predictions and len(span_predictions) != len(candidate_features):
                raise RuntimeError("span-confidence/cheap-control candidate alignment mismatch")
            if endpoint_predictions and span_predictions and any(
                left["candidate_interval_source_s"]
                != right["candidate_interval_source_s"]
                for left, right in zip(endpoint_predictions, span_predictions)
            ):
                raise RuntimeError("dual-confidence candidate alignment mismatch")

            raw_candidate_intervals = [
                [float(interval[0]), float(interval[1])]
                for interval in (parsed.intervals if parsed.parse_valid else [])
            ]
            raw_endpoint_predictions = endpoint_predictions
            raw_span_predictions = span_predictions
            raw_candidate_features = candidate_features
            keep_indices: list[int] = []
            seen_intervals: set[tuple[float, float]] = set()
            for candidate_index, interval in enumerate(raw_candidate_intervals):
                key = interval_key(interval)
                if args.exact_dedup and key in seen_intervals:
                    continue
                seen_intervals.add(key)
                keep_indices.append(candidate_index)

            candidate_intervals = [
                raw_candidate_intervals[index] for index in keep_indices
            ]

            def retain_and_reindex(
                values: Sequence[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                retained: list[dict[str, Any]] = []
                for new_index, source_index in enumerate(keep_indices):
                    item = dict(values[source_index])
                    item["span_index"] = new_index
                    retained.append(item)
                return retained

            endpoint_predictions = (
                retain_and_reindex(raw_endpoint_predictions)
                if raw_endpoint_predictions
                else []
            )
            span_predictions = (
                retain_and_reindex(raw_span_predictions)
                if raw_span_predictions
                else []
            )
            candidate_features = (
                retain_and_reindex(raw_candidate_features)
                if raw_candidate_features
                else []
            )
            total_candidate_count = len(raw_candidate_intervals)
            unique_candidate_count = len(
                {interval_key(interval) for interval in raw_candidate_intervals}
            )
            duplicate_candidate_count = (
                total_candidate_count - unique_candidate_count
            )
            token_alignment_statuses = sorted(
                {
                    str(item["token_alignment_status"])
                    for values in (
                        raw_endpoint_predictions,
                        raw_span_predictions,
                        raw_candidate_features,
                    )
                    for item in (values or [])
                    if item.get("token_alignment_status") is not None
                }
            )
            token_alignment_contracts = sorted(
                {
                    str(item["token_alignment_contract"])
                    for values in (
                        raw_endpoint_predictions,
                        raw_span_predictions,
                        raw_candidate_features,
                    )
                    for item in (values or [])
                    if item.get("token_alignment_contract") is not None
                }
            )
            if len(token_alignment_statuses) > 1 or len(token_alignment_contracts) > 1:
                raise RuntimeError("dual-head token alignment contract mismatch")
            hit_effective_max_new_tokens = bool(
                args.max_new_tokens is not None
                and len(completion_ids) >= args.max_new_tokens
            )
            truncated_output = bool(
                interval_parse_mode == "truncated_prefix_salvage"
                or (
                    hit_effective_max_new_tokens
                    and not parsed.source_parse.parse_valid
                    and not (
                        candidate_stopper is not None
                        and candidate_stopper.forced_close
                    )
                )
            )
            result = {
                **row,
                "schema_version": 1,
                "run_id": run_id,
                "arm": args.arm,
                "execution_position": position,
                "infer_status": "success",
                "source_model_revision": row.get("model_revision"),
                "model_revision": args.model_revision,
                "response": response,
                "completion_token_ids": completion_ids,
                "parse_valid": bool(parsed.parse_valid),
                "parse_error_code": (
                    None if parsed.parse_valid else parsed.source_parse.parse_error_code
                ),
                "source_parse_error_code": parsed.source_parse.parse_error_code,
                "strict_parse_valid": bool(parsed.source_parse.parse_valid),
                "normalization_status": parsed.normalization_status,
                "normalization_applied": parsed.normalization_applied,
                "candidate_intervals": candidate_intervals,
                "candidate_intervals_before_exact_dedup": raw_candidate_intervals,
                "endpoint_predictions": endpoint_predictions,
                "endpoint_predictions_before_exact_dedup": raw_endpoint_predictions,
                "span_predictions": span_predictions,
                "span_predictions_before_exact_dedup": raw_span_predictions,
                "generated_candidate_confidences": [
                    prediction["generated_confidence"]
                    for prediction in endpoint_predictions
                    if "generated_confidence" in prediction
                ],
                "fixed_candidate_diagnostics": fixed_diagnostics,
                "fixed16_diagnostics": (
                    fixed_diagnostics if fixed_candidate_count == 16 else None
                ),
                "fixed32_diagnostics": (
                    fixed_diagnostics if fixed_candidate_count == 32 else None
                ),
                "explicit_confidence_parse_mode": explicit_confidence_parse_mode,
                "interval_parse_mode": interval_parse_mode,
                "candidate_features": candidate_features,
                "candidate_features_before_exact_dedup": raw_candidate_features,
                "total_candidate_count": total_candidate_count,
                "unique_candidate_count": unique_candidate_count,
                "duplicate_candidate_count": duplicate_candidate_count,
                "duplicate_rate": (
                    duplicate_candidate_count / total_candidate_count
                    if total_candidate_count
                    else 0.0
                ),
                "token_alignment_status": (
                    token_alignment_statuses[0] if token_alignment_statuses else None
                ),
                "token_alignment_contract": (
                    token_alignment_contracts[0]
                    if token_alignment_contracts
                    else None
                ),
                "exact_dedup_applied": args.exact_dedup,
                "completion_tokens": len(completion_ids),
                "hit_effective_max_new_tokens": hit_effective_max_new_tokens,
                "truncated_output": truncated_output,
                "candidate_stop_reason": (
                    candidate_stopper.stop_reason
                    if candidate_stopper is not None
                    else None
                ),
                "candidate_stop_forced_close": bool(
                    candidate_stopper is not None
                    and candidate_stopper.forced_close
                ),
                "sampling_seed": row_sampling_seed,
                "generation_selected_logprobs": selected_logprobs,
                "generation_policy_mean_logprob": statistics.fmean(
                    float(item["selected_logprob"]) for item in decode_stats
                )
                if decode_stats
                else None,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "processor_contract": processor_contract,
                "single_pass_contract": {
                    "video_encodes": 1,
                    "autoregressive_decodes": 1,
                    "teacher_forced_replays": 0,
                    "vision_encode_forward_calls": vision_encode_calls,
                    "decode_forward_calls": len(call_states),
                    "confidence_state_source": (
                        "same_decode_yes_no_normalized_probability"
                        if fixed_hungarian_confidence
                        else "same_decode_explicit_json"
                        if args.explicit_candidate_confidence
                        else "same_decode_final_norm"
                        if head is not None or span_head is not None
                        else None
                    ),
                    "confidence_endpoint_anchor": args.endpoint_anchor
                    if head is not None
                    else None,
                    "span_confidence_state_source": (
                        "same_decode_final_layer_exact_span_hidden_mean"
                        if span_head is not None
                        else None
                    ),
                    "span_confidence_target": (
                        span_head_config.get("target") if span_head is not None else None
                    ),
                    "cheap_control_feature_anchor": "after_span",
                },
                "timing": {
                    "preprocessing_seconds": preprocessing_elapsed,
                    "generation_and_confidence_seconds": elapsed,
                    "end_to_end_seconds": time.perf_counter()
                    - end_to_end_started,
                    "pre_row_memory_allocated_bytes": pre_row_memory_allocated,
                    "pre_row_memory_reserved_bytes": pre_row_memory_reserved,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                },
                "paper_main_eligible": args.paper_main_eligible,
            }
            success += 1
            canonical_valid += int(parsed.parse_valid)
            strict_valid += int(parsed.source_parse.parse_valid)
            duration_filtered += int(parsed.normalization_applied)
        except Exception as exc:
            post_generation_failure = response is not None and completion_ids is not None
            result = {
                **row,
                "schema_version": 1,
                "run_id": run_id,
                "arm": args.arm,
                "execution_position": position,
                "infer_status": "fail",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "post_generation_failure": post_generation_failure,
                "response_before_failure": response if post_generation_failure else None,
                "completion_token_ids_before_failure": (
                    completion_ids if post_generation_failure else None
                ),
                "interval_parse_mode_before_failure": interval_parse_mode,
                "parse_valid_before_failure": (
                    bool(parsed.parse_valid) if parsed is not None else None
                ),
                "normalization_status_before_failure": (
                    parsed.normalization_status if parsed is not None else None
                ),
                "candidate_intervals_before_failure": (
                    parsed.intervals
                    if parsed is not None and parsed.parse_valid
                    else None
                ),
                "candidate_stop_reason_before_failure": (
                    candidate_stopper.stop_reason
                    if candidate_stopper is not None
                    else None
                ),
                "candidate_stop_forced_close_before_failure": bool(
                    candidate_stopper is not None and candidate_stopper.forced_close
                ),
                "paper_main_eligible": args.paper_main_eligible,
            }
            failed += 1
        append(output_path, result)
        progress = {
            "expected_rows": len(rows),
            "completed_rows": success + failed,
            "success_rows": success,
            "failed_rows": failed,
            "canonical_parse_rows": canonical_valid,
            "strict_parse_rows": strict_valid,
            "duration_filtered_rows": duration_filtered,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_json(args.output_dir / "progress.json", progress)
        print(canonical_json(progress), flush=True)
        # These names can retain GPU score/KV tensors or large CPU video/hidden
        # buffers until the next loop assignment.  Clear them before the next
        # video's preprocessing rather than relying on reassignment timing.
        prepared = rendered = processor_contract = generated = None
        candidate_stopper = call_states = completion_ids = step_scores = None
        response = parsed = endpoint_predictions = span_predictions = None
        interval_parse_mode = explicit_confidence_parse_mode = None
        decode_stats = selected_logprobs = feature_parse = candidate_features = None
        raw_candidate_intervals = raw_endpoint_predictions = raw_span_predictions = None
        result = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        write_json(
            args.output_dir / "memory-progress.json",
            {
                "execution_position": position,
                "post_cleanup_memory_allocated_bytes": int(
                    torch.cuda.memory_allocated(device)
                ),
                "post_cleanup_memory_reserved_bytes": int(
                    torch.cuda.memory_reserved(device)
                ),
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )

    completion = {
        "expected_rows": len(rows),
        "completed_rows": success + failed,
        "success_rows": success,
        "failed_rows": failed,
        "canonical_parse_rows": canonical_valid,
        "strict_parse_rows": strict_valid,
        "duration_filtered_rows": duration_filtered,
    }
    manifest["status"] = "success" if failed == 0 and success == len(rows) else "complete_with_row_failures"
    manifest["completion"] = completion
    write_json(args.output_dir / "run-manifest.json", manifest)
    write_json(args.output_dir / "_COMPLETE", completion)
    if failed == 0 and success == len(rows):
        write_json(args.output_dir / "_SUCCESS", completion)
    return 0 if success + failed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
