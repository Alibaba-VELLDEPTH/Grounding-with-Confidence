"""JSON interval adapter for the two predeclared v1.3 GRPO reward arms."""
import json
import math
import os

from tvg_confidence.intervals import canonicalize_interval_set_to_duration, single_span_iou, set_tiou

NMS = 0.3
THRESHOLD = 0.18276141583919525
MAX_CANDIDATES = 32
MAX_SOURCE_SPANS = 256


def pool(text, duration):
    parsed = canonicalize_interval_set_to_duration(text, duration)
    unique, source_indices, seen = [], [], set()
    for span, index in zip(parsed.intervals, parsed.source_span_indices):
        key = tuple(span)
        if key not in seen and len(unique) < MAX_CANDIDATES:
            unique.append(span)
            source_indices.append(index)
            seen.add(key)
    keep = []
    for index, span in enumerate(unique):
        if all(single_span_iou(span, unique[other]) <= NMS for other in keep):
            keep.append(index)
    return parsed, unique, [unique[i] for i in keep], [source_indices[i] for i in keep]


def matched_count(predictions, targets, threshold=0.5):
    """Maximum-cardinality one-to-one occurrence matching; one span cannot cover two GTs."""
    edges = [[j for j, target in enumerate(targets)
              if single_span_iou(span, target) >= threshold] for span in predictions]
    owners = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or augment(owners[j], visited):
                owners[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(predictions)))


def batch_reward(data_sources, solution_strs, ground_truths, extra_infos,
                 valid_response_lengths=None, **kwargs):
    arm = os.environ["TVG_RL_ARM"]
    if arm not in {"temporal", "recall_head"}:
        raise ValueError("unknown reward arm")
    results = []
    for i, (text, truth, extra) in enumerate(zip(solution_strs, ground_truths, extra_infos, strict=True)):
        gt = json.loads(truth)
        if not gt or any(not 0 <= a < b <= extra["duration_s"] + 1e-4 for a, b in gt):
            raise ValueError("invalid RL source ground truth")
        parsed, unique, nms, indices = pool(text, extra["duration_s"])
        scores = extra["tvg_span_scores"]
        if len(scores) != MAX_SOURCE_SPANS:
            raise RuntimeError("missing actor-forward head scores")
        survivor_scores = [float(scores[j]) for j in indices]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in survivor_scores):
            raise RuntimeError("head score/parsed source index misalignment")
        selected = [span for span, score in zip(nms, survivor_scores) if score >= THRESHOLD]
        recall = matched_count(nms, gt) / len(gt)
        tiou = set_tiou(nms, gt)
        selected_tiou = set_tiou(selected, gt)
        cacc = float(len(nms) == len(gt))
        reward = 0.5 * (tiou + cacc) if arm == "temporal" else 0.5 * (recall + selected_tiou)
        if not parsed.parse_valid:
            reward = 0.0
        match = matched_count(selected, gt)
        selected_f1 = 2 * match / (len(selected) + len(gt))
        labels = [float(max(single_span_iou(span, target) for target in gt) >= 0.5) for span in nms]
        clipped = [min(1 - 1e-7, max(1e-7, s)) for s in survivor_scores]
        bce = sum(-y * math.log(s) - (1 - y) * math.log(1 - s) for s, y in zip(clipped, labels))
        brier = sum((s - y) ** 2 for s, y in zip(survivor_scores, labels))
        results.append({"score": reward, "pool_recall05": recall, "nms_tiou": tiou,
            "nms_cacc": cacc, "selected_tiou": selected_tiou, "selected_f1_05": selected_f1,
            "selected_recall05": match / len(gt), "raw_count": len(parsed.source_parse.intervals),
            "unique_count": len(unique), "nms_count": len(nms), "selected_count": len(selected),
            "invalid": float(not parsed.parse_valid), "completion_at_cap": float(
                valid_response_lengths is not None and int(valid_response_lengths[i]) >= 512),
            "head_mean": sum(survivor_scores) / max(1, len(survivor_scores)),
            "head_keep_fraction": len(selected) / max(1, len(nms)),
            "head_bce": bce / max(1, len(nms)), "head_brier": brier / max(1, len(nms)),
            "head_positive_rate": sum(labels) / max(1, len(nms)),
            "head_false_negative_count": sum(y == 1 and s < THRESHOLD for s, y in zip(survivor_scores, labels)),
            "head_false_positive_count": sum(y == 0 and s >= THRESHOLD for s, y in zip(survivor_scores, labels))})
    return results
