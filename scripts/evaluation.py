"""Query-level temporal grounding metrics used by the paper."""
from __future__ import annotations
import statistics
from typing import Sequence
from tvg_confidence.intervals import set_tiou, single_span_iou
from tvg_confidence.matching import maximum_weight_assignment

THRESHOLDS = (0.3, 0.5, 0.7)

def official_multispan_iou(
    prediction: Sequence[Sequence[float]],
    ground_truth: Sequence[Sequence[float]],
) -> float:
    """Match TimeLens2/VUE-TR's released multi-span mIoU exactly.

    The official evaluator merges overlapping *predictions* but leaves the
    annotation list as supplied.  Keeping this implementation separate from
    ``set_tiou`` prevents a future dataset with overlapping GT spans from
    silently changing the benchmark metric.
    """

    if not ground_truth:
        return 1.0 if not prediction else 0.0
    if not prediction:
        return 0.0
    ordered = sorted(
        ([float(span[0]), float(span[1])] for span in prediction if len(span) >= 2),
        key=lambda span: (span[0], span[1]),
    )
    if not ordered:
        return 0.0
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    merged = [span for span in merged if span[0] <= span[1]]
    if not merged:
        return 0.0
    gt = [
        [float(span[0]), float(span[1])]
        for span in ground_truth
        if len(span) >= 2
    ]
    prediction_length = sum(end - start for start, end in merged)
    ground_truth_length = sum(end - start for start, end in gt)
    intersection = sum(
        max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
        for pred_start, pred_end in merged
        for gt_start, gt_end in gt
    )
    union = prediction_length + ground_truth_length - intersection
    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, intersection / (union + 1e-16)))

def set_metrics(prediction: Sequence[Sequence[float]], ground_truth: Sequence[Sequence[float]]) -> dict[str, float]:
    matrix = [
        [single_span_iou(candidate, target) for target in ground_truth]
        for candidate in prediction
    ]
    matches = [
        float(matrix[left][right])
        for left, right in maximum_weight_assignment(matrix)
    ] if matrix else []
    set_overlap = float(set_tiou(prediction, ground_truth))
    official_overlap = official_multispan_iou(prediction, ground_truth)
    output: dict[str, float] = {
        "predicted_count": float(len(prediction)),
        "absolute_count_error": float(abs(len(prediction) - len(ground_truth))),
        "count_accuracy": float(len(prediction) == len(ground_truth)),
        "set_tiou": set_overlap,
        "miou": official_overlap,
    }
    f1s = []
    for threshold in THRESHOLDS:
        true_positive = sum(value >= threshold for value in matches)
        precision = true_positive / len(prediction) if prediction else 0.0
        recall = true_positive / len(ground_truth) if ground_truth else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        tag = str(threshold).replace(".", "")
        output[f"occurrence_precision_{tag}"] = precision
        output[f"occurrence_recall_{tag}"] = recall
        output[f"occurrence_f1_{tag}"] = f1
        f1s.append(f1)
    output["etf1"] = output["count_accuracy"] * statistics.fmean(f1s)
    return output
