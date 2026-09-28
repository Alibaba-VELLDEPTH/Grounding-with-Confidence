#!/usr/bin/env python3
"""Evaluate a frozen threshold on saved candidates without model dependencies."""
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
from evaluation import set_metrics

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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Fixed confidence threshold; no tuning on these evaluation rows")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be finite and within [0, 1]")
    per_query, seen = [], set()
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row["request_id"])
        if uid in seen:
            raise ValueError("duplicate request_id")
        seen.add(uid)
        duration = float(row["source_duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid video duration")
        gt = spans(row["gt_intervals"], duration, "ground truth")
        failed = row.get("infer_status", "success") != "success"
        if failed:
            selected = []
        else:
            candidates = spans(row["candidate_intervals"], duration, "candidate")
            raw_scores = row.get("span_scores")
            if raw_scores is None:
                raw_scores = [item["predicted_span_correctness_probability"]
                              for item in row.get("span_predictions", [])]
            scores = list(map(float, raw_scores))
            if len(candidates) != len(scores) or any(
                    not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
                raise ValueError("candidate/score alignment or range mismatch")
            parsed, _, survivors, indices = pool(json.dumps(candidates), duration)
            if candidates and not parsed.parse_valid:
                raise ValueError("invalid candidate sequence")
            selected = [span for span, index in zip(survivors, indices)
                        if scores[index] >= args.threshold]
        per_query.append({"request_id": uid, "inference_failed": failed,
                          "selected_intervals": selected, "metrics": set_metrics(selected, gt)})
    if not per_query:
        raise ValueError("no evaluation rows")
    report = {"queries": len(per_query), "failed_queries": sum(r["inference_failed"] for r in per_query),
              "threshold": args.threshold, "nms_iou": 0.3,
              "averaging": "query macro; failed queries remain in the denominator",
              "metrics": {key: statistics.fmean(r["metrics"][key] for r in per_query)
                          for key in per_query[0]["metrics"]}, "per_query": per_query}
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end="")

if __name__ == "__main__":
    main()
