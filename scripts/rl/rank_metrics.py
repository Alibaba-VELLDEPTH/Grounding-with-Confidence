"""Ranked-coverage diagnostic used by the main reward implementation."""
from collections import defaultdict
import math
from reward import matched_count

def coverage_ap(spans, scores, ground_truth, iou_threshold=0.5):
    """Area sum P_k * delta R_k at realizable confidence thresholds.

    Prefix TP uses maximum-cardinality one-to-one matching. Equal scores enter
    together, so neither generation order within a tie nor score calibration
    supplies an artificial ranking benefit. The denominator includes missed GT.
    """
    if len(spans) != len(scores) or not ground_truth:
        raise ValueError("span/score alignment and nonempty GT are required")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("invalid confidence")
    groups = defaultdict(list)
    for span, score in zip(spans, scores, strict=True):
        groups[score].append(span)
    prefix, previous, total = [], 0, 0.0
    for score in sorted(groups, reverse=True):
        prefix.extend(groups[score])
        matched = matched_count(prefix, ground_truth, iou_threshold)
        total += (matched - previous) / len(ground_truth) * matched / len(prefix)
        previous = matched
    return total
