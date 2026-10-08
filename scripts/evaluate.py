#!/usr/bin/env python3
"""Evaluate saved candidates at a fixed threshold without model dependencies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tvg_confidence.selection import pool
from evaluation import omtg_metrics, set_metrics


def spans(value, duration, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = []
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in pair)):
            raise ValueError(f"invalid {field} interval")
        a, b = map(float, pair)
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration):
            raise ValueError(f"{field} interval outside the video")
        result.append([a, b])
    return result


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed_rows(rows):
    indexed = {}
    for row in rows:
        uid = str(row["request_id"])
        if uid in indexed:
            raise ValueError(f"duplicate request_id: {uid}")
        indexed[uid] = row
    if not indexed:
        raise ValueError("no evaluation rows")
    return indexed


def select_intervals(row, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    if row.get("infer_status", "success") != "success":
        return []
    duration = float(row["source_duration_s"])
    candidates = spans(row["candidate_intervals"], duration, "candidate")
    raw_scores = row.get("span_scores")
    if raw_scores is None:
        raw_scores = [item["predicted_span_correctness_probability"]
                      for item in row.get("span_predictions", [])]
    if not isinstance(raw_scores, list) or any(isinstance(v, bool) for v in raw_scores):
        raise ValueError("scores must be a list of probabilities")
    scores = list(map(float, raw_scores))
    if len(candidates) != len(scores) or any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError("candidate/score alignment or range mismatch")
    parsed, _, survivors, indices = pool(json.dumps(candidates), duration)
    if candidates and not parsed.parse_valid:
        raise ValueError("invalid candidate sequence")
    return [span for span, index in zip(survivors, indices) if scores[index] >= threshold]


def evaluate_rows(rows, threshold, requests=None, metric_protocol="omtg"):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    if metric_protocol not in {"omtg", "set"}:
        raise ValueError("metric protocol must be omtg or set")
    metric_function = omtg_metrics if metric_protocol == "omtg" else set_metrics
    predictions = indexed_rows(rows)
    expected = indexed_rows(requests) if requests is not None else None
    if expected is not None and predictions.keys() != expected.keys():
        raise ValueError("prediction/request ID mismatch; retain failed queries as explicit rows")
    per_query = []
    for uid, row in predictions.items():
        annotation = expected[uid] if expected is not None else row
        duration = float(annotation["source_duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid video duration")
        if float(row["source_duration_s"]) != duration:
            raise ValueError(f"prediction/request duration mismatch: {uid}")
        gt = spans(annotation["gt_intervals"], duration, "ground truth")
        if "gt_intervals" in row and spans(row["gt_intervals"], duration, "ground truth") != gt:
            raise ValueError(f"prediction/request ground truth mismatch: {uid}")
        selected = select_intervals(row, threshold)
        per_query.append({"request_id": uid,
                          "inference_failed": row.get("infer_status", "success") != "success",
                          "selected_intervals": selected, "metrics": metric_function(selected, gt)})
    return {"queries": len(per_query),
            "failed_queries": sum(r["inference_failed"] for r in per_query),
            "threshold": threshold, "nms_iou": 0.3, "metric_protocol": metric_protocol,
            "averaging": "query macro; failed queries remain in the denominator",
            "request_identity_verified": expected is not None,
            "metrics": {key: statistics.fmean(r["metrics"][key] for r in per_query)
                        for key in per_query[0]["metrics"]}, "per_query": per_query}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--requests", type=Path,
                        help="Original requests with GT; require exactly the full prediction denominator")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Fixed confidence threshold; no tuning on these evaluation rows")
    parser.add_argument("--metric-protocol", choices=("omtg", "set"), default="omtg",
                        help="Official OMTG parity (default), or legacy dependency-free set metrics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_rows(read_rows(args.input), args.threshold,
                           read_rows(args.requests) if args.requests else None, args.metric_protocol)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
