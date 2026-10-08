"""Recall plus selected F1 at one threshold shared by the whole TRAIN batch."""
from collections import defaultdict
import json
import math
import os

from reward import MAX_SOURCE_SPANS, batch_reward as baseline_batch_reward, matched_count, pool
from rank_metrics import coverage_ap
from tvg_confidence.intervals import set_tiou

VARIANTS = {"recall_batch_global_f1", "recall_batch_global_f1_tiou"}


def shared_threshold(pools, scores, targets, valid=None):
    """Exact macro-F1 sweep; update only queries affected by each score group.

    A common monotone transformation preserves the selected sets. Independent
    transformations per query can change the objective. Matching is recomputed
    after additions so an ambiguous earlier candidate can be reassigned.
    """
    if not pools or not len(pools) == len(scores) == len(targets):
        raise ValueError("nonempty aligned batch required")
    valid = [True] * len(pools) if valid is None else valid
    if len(valid) != len(pools):
        raise ValueError("validity alignment failed")
    events = defaultdict(lambda: defaultdict(list))
    for i, (spans, confidence, gt) in enumerate(zip(pools, scores, targets, strict=True)):
        if len(spans) != len(confidence) or not gt:
            raise ValueError("span scores or GT misaligned")
        for span, score in zip(spans, confidence, strict=True):
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("nonfinite or out-of-range confidence")
            events[score][i].append(span)
    # Include both all-drop and all-keep. Equal score candidates enter together.
    initial = math.nextafter(max(events), math.inf) if events else 0.5
    best = {"threshold": initial, "macro_f1": 0.0, "macro_recall": 0.0}
    prefixes = [[] for _ in pools]
    f1s, recalls = [0.0] * len(pools), [0.0] * len(pools)
    total_f1 = total_recall = 0.0
    events[0.0]  # all-keep endpoint, including an entirely empty batch
    for threshold in sorted(events, reverse=True):
        for i, additions in events[threshold].items():
            prefixes[i].extend(additions)
            tp = matched_count(prefixes[i], targets[i]) if valid[i] else 0
            f1 = 2 * tp / (len(prefixes[i]) + len(targets[i]))
            recall = tp / len(targets[i])
            total_f1 += f1 - f1s[i]
            total_recall += recall - recalls[i]
            f1s[i], recalls[i] = f1, recall
        mean_f1, mean_recall = total_f1 / len(pools), total_recall / len(pools)
        better = mean_f1 > best["macro_f1"] + 1e-12
        tied_f1 = abs(mean_f1 - best["macro_f1"]) <= 1e-12
        better_recall = mean_recall > best["macro_recall"] + 1e-12
        tied_recall = abs(mean_recall - best["macro_recall"]) <= 1e-12
        if better or tied_f1 and (better_recall or tied_recall and threshold < best["threshold"]):
            best = {"threshold": threshold, "macro_f1": mean_f1, "macro_recall": mean_recall}
    return {**best, "distinct_score_groups": len(events), "batch_size": len(pools)}


def batch_reward(data_sources, solution_strs, ground_truths, extra_infos,
                 valid_response_lengths=None, **kwargs):
    variant = os.environ["TVG_RL_REWARD_VARIANT"]
    if variant not in VARIANTS:
        raise ValueError("wrong shared-threshold reward variant")
    results = baseline_batch_reward(data_sources, solution_strs, ground_truths, extra_infos,
        valid_response_lengths=valid_response_lengths, **kwargs)
    targets = [json.loads(truth) for truth in ground_truths]
    parsed = [pool(text, extra["duration_s"]) for text, extra in zip(solution_strs, extra_infos, strict=True)]
    pools = [item[2] for item in parsed]
    scores = [[float(extra["tvg_span_scores"][i]) for i in item[3]]
        for item, extra in zip(parsed, extra_infos, strict=True)]
    valid = [item[0].parse_valid for item in parsed]
    sweep = shared_threshold(pools, scores, targets, valid)
    threshold = sweep["threshold"]
    for i, result in enumerate(results):
        for name in ("head_false_negative_count", "head_false_positive_count"):
            result[name + "_legacy_fixed"] = result.pop(name)
        spans, confidence, gt = pools[i], scores[i], targets[i]
        selected = [span for span, score in zip(spans, confidence, strict=True) if score >= threshold]
        matched = matched_count(selected, gt) if valid[i] else 0
        f1 = 2 * matched / (len(selected) + len(gt))
        tiou = set_tiou(selected, gt) if valid[i] else 0.0
        with_tiou = variant == "recall_batch_global_f1_tiou"
        quality = 0.5 * (f1 + tiou) if with_tiou else f1
        result.update(score=0.5 * (result["pool_recall05"] + quality) if valid[i] else 0.0,
            quality_reward=quality, selected_f1_05=f1, selected_recall05=matched / len(gt),
            selected_count=len(selected), selected_tiou=tiou,
            selected_matched_gt=matched, selected_unmatched_count=len(selected) - matched,
            selected_missed_gt=len(gt) - matched, head_keep_fraction=len(selected) / max(1, len(spans)),
            global_threshold=threshold, global_threshold_batch_size=sweep["batch_size"],
            global_threshold_macro_f1=sweep["macro_f1"],
            global_threshold_macro_recall=sweep["macro_recall"],
            global_threshold_distinct_score_groups=sweep["distinct_score_groups"],
            recall_reward_contribution=0.5 * result["pool_recall05"] if valid[i] else 0.0,
            quality_reward_contribution=0.5 * quality,
            selected_f1_reward_contribution=(0.25 if with_tiou else 0.5) * f1,
            selected_tiou_reward_contribution=0.25 * tiou if with_tiou else 0.0,
            gt_count=len(gt),
            raw_unique_matched_gt=matched_count(parsed[i][1], gt), nms_matched_gt=matched_count(spans, gt),
            nms_lost_gt=matched_count(parsed[i][1], gt) - matched_count(spans, gt),
            rank_ap05=coverage_ap(spans, confidence, gt),
            exact_duplicate_count=len(parsed[i][0].source_parse.intervals) - len({tuple(s) for s in parsed[i][0].source_parse.intervals}),
            query_id=str(extra_infos[i].get("query_id", "")),
            nms_spans_json=json.dumps(spans), nms_scores_json=json.dumps(confidence))
    return results


def training_diagnostics(extras, prompt_ids):
    metrics = {"reward/" + name + "_mean": sum(values) / len(values)
        for name, values in extras.items()
        if len(values) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)}
    total_gt, total_nms = sum(extras["gt_count"]), sum(extras["nms_count"])
    for name in ("raw_unique", "nms", "selected"):
        metrics["reward/" + name + "_recall05_micro"] = sum(extras[name + "_matched_gt"]) / max(1, total_gt)
    metrics["reward/selected_keep_fraction_micro"] = sum(extras["selected_count"]) / max(1, total_nms)
    if len(set(extras["global_threshold"])) != 1:
        raise ValueError("training reward must use ONE shared threshold per batch")
    if prompt_ids is not None:
        if len(prompt_ids) != len(extras["score"]):
            raise ValueError("prompt/reward alignment failed")
        for component in ("score", "pool_recall05", "quality_reward", "selected_f1_05", "selected_tiou", "rank_ap05", "nms_tiou"):
            groups = defaultdict(list)
            for uid, value in zip(prompt_ids, extras[component], strict=True):
                groups[str(uid)].append(value)
            stds = [(sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5 for v in groups.values()]
            metrics["reward/" + component + "_group_std_mean"] = sum(stds) / max(1, len(stds))
            metrics["reward/" + component + "_constant_group_fraction"] = sum(s < 1e-12 for s in stds) / max(1, len(stds))
        metrics["reward/zero_advantage_group_fraction"] = metrics["reward/score_constant_group_fraction"]
        metrics["reward/prompt_groups"] = len(groups)
        metrics["reward/rollouts_per_prompt_min"] = min(map(len, groups.values()))
        metrics["reward/rollouts_per_prompt_max"] = max(map(len, groups.values()))
    return metrics
