from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


JSON_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
TIME_TEXT_RE = re.compile(
    r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
)


@dataclass(frozen=True)
class CharRegion:
    char_start: int
    char_end: int
    semantic_role: str
    span_index: Optional[int] = None
    value_piece_index: Optional[int] = None


@dataclass(frozen=True)
class SemanticAnchor:
    semantic_event: str
    char_position: int
    side: str
    span_index: Optional[int] = None


@dataclass
class IntervalParse:
    raw_text: str
    payload_text: Optional[str]
    payload_char_start: Optional[int]
    payload_char_end: Optional[int]
    intervals: List[List[float]]
    parse_valid: bool
    syntax_valid: bool
    parse_error_code: Optional[str]
    parse_error_detail: Optional[str]
    char_regions: List[CharRegion] = field(default_factory=list)
    semantic_anchors: List[SemanticAnchor] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["char_regions"] = [asdict(region) for region in self.char_regions]
        value["semantic_anchors"] = [
            asdict(anchor) for anchor in self.semantic_anchors
        ]
        return value


@dataclass
class CanonicalIntervalSet:
    """Deployment-time interval set plus its source-token correspondence.

    ``source_span_indices`` keeps deterministic post-processing aligned with
    the generated closing-token states.  It is especially important when an
    out-of-range suffix collapses at the video boundary and is dropped.
    """

    source_parse: IntervalParse
    intervals: List[List[float]]
    source_span_indices: List[int]
    parse_valid: bool
    normalization_status: str

    @property
    def normalization_applied(self) -> bool:
        return self.normalization_status == "duration_filter"

    @property
    def parse_error_code(self) -> Optional[str]:
        return self.source_parse.parse_error_code


def _invalid(
    raw_text: str,
    code: str,
    detail: str,
    payload_text: Optional[str] = None,
    payload_start: Optional[int] = None,
    intervals: Optional[List[List[float]]] = None,
    *,
    syntax_valid: bool = False,
    char_regions: Optional[List[CharRegion]] = None,
    semantic_anchors: Optional[List[SemanticAnchor]] = None,
) -> IntervalParse:
    return IntervalParse(
        raw_text=raw_text,
        payload_text=payload_text,
        payload_char_start=payload_start,
        payload_char_end=(
            payload_start + len(payload_text)
            if payload_text is not None and payload_start is not None
            else None
        ),
        intervals=intervals or [],
        parse_valid=False,
        syntax_valid=syntax_valid,
        parse_error_code=code,
        parse_error_detail=detail,
        char_regions=char_regions or [],
        semantic_anchors=semantic_anchors or [],
    )


def _candidate_json_arrays(text: str) -> List[Tuple[int, int, str, Any]]:
    decoder = json.JSONDecoder()
    candidates: List[Tuple[int, int, str, Any]] = []
    seen = set()
    for index, character in enumerate(text):
        if character != "[":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list):
            continue
        end = index + consumed
        key = (index, end)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((index, end, text[index:end], value))
    return candidates


def _validate_interval_root(value: Any) -> Tuple[Optional[str], List[List[float]]]:
    if not isinstance(value, list):
        return "non_list_root", []
    intervals: List[List[float]] = []
    for span in value:
        if not isinstance(span, list) or len(span) != 2:
            return "non_pair_span", intervals
        if (
            isinstance(span[0], bool)
            or isinstance(span[1], bool)
            or not isinstance(span[0], (int, float))
            or not isinstance(span[1], (int, float))
        ):
            return "non_numeric_boundary", intervals
        start, end = float(span[0]), float(span[1])
        if not (math.isfinite(start) and math.isfinite(end)):
            return "non_finite_boundary", intervals
        intervals.append([start, end])
    return None, intervals


def _find_pair_brackets(
    payload: str, number_spans: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int, int, int, int, int, int]]:
    pairs = []
    for pair_index in range(len(number_spans) // 2):
        start_number = number_spans[2 * pair_index]
        end_number = number_spans[2 * pair_index + 1]
        open_index = payload.rfind("[", 0, start_number[0] + 1)
        close_index = payload.find("]", end_number[1])
        delimiter = payload.find(",", start_number[1], end_number[0])
        if open_index < 0 or close_index < 0 or delimiter < 0:
            raise ValueError(f"unable to locate JSON punctuation for span {pair_index}")
        pairs.append(
            (
                open_index,
                start_number[0],
                start_number[1],
                delimiter,
                end_number[0],
                end_number[1],
                close_index,
            )
        )
    return pairs


def _build_semantics(
    payload: str, payload_start: int, interval_count: int
) -> Tuple[List[CharRegion], List[SemanticAnchor]]:
    number_matches = list(JSON_NUMBER_RE.finditer(payload))
    if len(number_matches) != interval_count * 2:
        raise ValueError(
            f"expected {interval_count * 2} numeric spans, found {len(number_matches)}"
        )
    number_spans = [(match.start(), match.end()) for match in number_matches]
    pairs = _find_pair_brackets(payload, number_spans)
    root_open = payload.find("[")
    root_close = payload.rfind("]")
    if root_open < 0 or root_close < root_open:
        raise ValueError("unable to locate root JSON brackets")

    regions: List[CharRegion] = [
        CharRegion(payload_start + root_open, payload_start + root_open + 1, "set_open"),
        CharRegion(payload_start + root_close, payload_start + root_close + 1, "set_close"),
    ]
    anchors: List[SemanticAnchor] = []

    for span_index, pair in enumerate(pairs):
        (
            open_index,
            start_start,
            start_end,
            pair_delimiter,
            end_start,
            end_end,
            close_index,
        ) = pair
        regions.extend(
            [
                CharRegion(
                    payload_start + open_index,
                    payload_start + open_index + 1,
                    "span_open",
                    span_index,
                ),
                CharRegion(
                    payload_start + start_start,
                    payload_start + start_end,
                    "start_value",
                    span_index,
                    0,
                ),
                CharRegion(
                    payload_start + pair_delimiter,
                    payload_start + pair_delimiter + 1,
                    "pair_delimiter",
                    span_index,
                ),
                CharRegion(
                    payload_start + end_start,
                    payload_start + end_end,
                    "end_value",
                    span_index,
                    0,
                ),
                CharRegion(
                    payload_start + close_index,
                    payload_start + close_index + 1,
                    "span_close",
                    span_index,
                ),
            ]
        )
        anchors.extend(
            [
                SemanticAnchor(
                    "before_start", payload_start + start_start, "before", span_index
                ),
                SemanticAnchor(
                    "after_start", payload_start + start_end, "after", span_index
                ),
                SemanticAnchor(
                    "before_end", payload_start + end_start, "before", span_index
                ),
                SemanticAnchor(
                    "after_end", payload_start + end_end, "after", span_index
                ),
                SemanticAnchor(
                    "after_span", payload_start + close_index + 1, "after", span_index
                ),
            ]
        )

    for left, right in zip(pairs, pairs[1:]):
        delimiter = payload.find(",", left[-1] + 1, right[0])
        if delimiter >= 0:
            regions.append(
                CharRegion(
                    payload_start + delimiter,
                    payload_start + delimiter + 1,
                    "span_delimiter",
                )
            )

    anchors.extend(
        [
            SemanticAnchor(
                "before_set_end", payload_start + root_close, "before", None
            ),
            SemanticAnchor("after_set", payload_start + root_close + 1, "after", None),
        ]
    )
    return sorted(regions, key=lambda item: (item.char_start, item.char_end)), anchors


def parse_interval_set_strict(
    text: str, duration_s: Optional[float] = None
) -> IntervalParse:
    raw_text = "" if text is None else str(text)
    if not raw_text.strip():
        return _invalid(raw_text, "empty_output", "output is empty")

    raw_candidates = _candidate_json_arrays(raw_text)
    valid_candidates = []
    structural_errors = []
    for start, end, payload, value in raw_candidates:
        error, intervals = _validate_interval_root(value)
        if error is None:
            valid_candidates.append((start, end, payload, intervals))
        else:
            structural_errors.append((error, start, payload, intervals))

    if len(valid_candidates) > 1:
        outer_candidates = []
        for candidate in valid_candidates:
            start, end = candidate[0], candidate[1]
            contains_other = any(
                start <= other[0] and end >= other[1] and (start, end) != (other[0], other[1])
                for other in valid_candidates
            )
            if contains_other:
                outer_candidates.append(candidate)
        if len(outer_candidates) == 1:
            valid_candidates = outer_candidates

    if len(valid_candidates) > 1:
        return _invalid(
            raw_text,
            "ambiguous_multiple_payloads",
            f"found {len(valid_candidates)} valid interval arrays",
        )
    if not valid_candidates:
        stripped = raw_text.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            code = "truncated_output" if stripped.count("[") > stripped.count("]") else "invalid_json"
            return _invalid(raw_text, code, str(exc))
        error, intervals = _validate_interval_root(parsed)
        if error is None:
            return _invalid(
                raw_text,
                "other",
                "valid root array was not recoverable with source offsets",
                payload_text=stripped,
                payload_start=raw_text.find(stripped),
                intervals=intervals,
            )
        return _invalid(
            raw_text,
            error,
            f"JSON root failed interval validation: {error}",
            payload_text=stripped,
            payload_start=raw_text.find(stripped),
            intervals=intervals,
        )

    start, end, payload, intervals = valid_candidates[0]
    try:
        regions, anchors = _build_semantics(payload, start, len(intervals))
    except ValueError as exc:
        return _invalid(
            raw_text,
            "semantic_token_alignment_failed",
            str(exc),
            payload,
            start,
            intervals,
        )

    for span_index, (span_start, span_end) in enumerate(intervals):
        if span_start >= span_end:
            return _invalid(
                raw_text,
                "reversed_boundary",
                f"span {span_index} has start >= end",
                payload,
                start,
                intervals,
                syntax_valid=True,
                char_regions=regions,
                semantic_anchors=anchors,
            )
        if duration_s is not None and (
            span_start < 0 or span_end > float(duration_s) + 1e-6
        ):
            return _invalid(
                raw_text,
                "out_of_range",
                f"span {span_index} is outside [0, {duration_s}]",
                payload,
                start,
                intervals,
                syntax_valid=True,
                char_regions=regions,
                semantic_anchors=anchors,
            )

    return IntervalParse(
        raw_text=raw_text,
        payload_text=payload,
        payload_char_start=start,
        payload_char_end=end,
        intervals=intervals,
        parse_valid=True,
        syntax_valid=True,
        parse_error_code=None,
        parse_error_detail=None,
        char_regions=regions,
        semantic_anchors=anchors,
    )


def canonicalize_interval_set_to_duration(
    text: str, duration_s: float
) -> CanonicalIntervalSet:
    """Project a syntactically complete interval set onto the video domain.

    A malformed JSON response remains invalid.  For a syntactically complete
    response whose only strict-parse failure is a boundary violation, retain
    only candidates that already satisfy ``0 <= start < end <= duration_s``.
    Boundaries are never clamped or swapped and missing syntax is never
    invented.  This is a model-agnostic deployment constraint, not a
    confidence decision.
    """

    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"duration_s must be finite and positive: {duration_s}")
    parsed = parse_interval_set_strict(text, duration_s=duration)
    if parsed.parse_valid:
        return CanonicalIntervalSet(
            source_parse=parsed,
            intervals=[list(map(float, span)) for span in parsed.intervals],
            source_span_indices=list(range(len(parsed.intervals))),
            parse_valid=True,
            normalization_status="strict",
        )
    if (
        not parsed.syntax_valid
        or parsed.parse_error_code not in {"out_of_range", "reversed_boundary"}
    ):
        return CanonicalIntervalSet(
            source_parse=parsed,
            intervals=[],
            source_span_indices=[],
            parse_valid=False,
            normalization_status="invalid",
        )

    intervals: List[List[float]] = []
    source_span_indices: List[int] = []
    for source_span_index, (raw_start, raw_end) in enumerate(parsed.intervals):
        start = float(raw_start)
        end = float(raw_end)
        if not (0.0 <= start < end <= duration):
            continue
        intervals.append([start, end])
        source_span_indices.append(source_span_index)
    return CanonicalIntervalSet(
        source_parse=parsed,
        intervals=intervals,
        source_span_indices=source_span_indices,
        parse_valid=bool(intervals),
        normalization_status="duration_filter" if intervals else "invalid",
    )


def merge_intervals(intervals: Sequence[Sequence[float]]) -> List[List[float]]:
    valid = sorted(
        ([float(span[0]), float(span[1])] for span in intervals if len(span) >= 2),
        key=lambda span: (span[0], span[1]),
    )
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def set_tiou(
    prediction: Sequence[Sequence[float]], ground_truth: Sequence[Sequence[float]]
) -> float:
    pred = merge_intervals(prediction)
    gt = merge_intervals(ground_truth)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    intersection = 0.0
    for pred_span in pred:
        for gt_span in gt:
            intersection += max(
                0.0,
                min(pred_span[1], gt_span[1]) - max(pred_span[0], gt_span[0]),
            )
    pred_length = sum(end - start for start, end in pred)
    gt_length = sum(end - start for start, end in gt)
    union = pred_length + gt_length - intersection
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def single_span_iou(a: Sequence[float], b: Sequence[float]) -> float:
    max_start = max(float(a[0]), float(b[0]))
    min_start = min(float(a[0]), float(b[0]))
    max_end = max(float(a[1]), float(b[1]))
    min_end = min(float(a[1]), float(b[1]))
    denominator = max_end - min_start
    if denominator <= 0:
        return 0.0
    return max(min_end - max_start, 0.0) / denominator


def extract_time_official(text: str) -> List[Tuple[float, float]]:
    paragraph = str(text).lower()
    timestamps: List[Tuple[float, float]] = []

    time_matches = TIME_TEXT_RE.findall(paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]
    if time_matches:
        converted = []
        for value in time_matches:
            parts = value.split(":")
            if len(parts) == 3:
                hours, minutes = map(int, parts[:2])
                converted.append(float(hours * 3600 + minutes * 60 + float(parts[2])))
            else:
                converted.append(float(int(parts[0]) * 60 + float(parts[1])))
        timestamps = [
            (converted[index], converted[index + 1])
            for index in range(0, len(converted), 2)
        ]

    if not timestamps:
        for pattern in (
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
        ):
            matches = re.findall(pattern, paragraph)
            if matches:
                timestamps = [(float(start), float(end)) for start, end in matches]
                break

    if not timestamps:
        values = re.findall(r"\b(\d+\.\d+|\d+)\b", paragraph)
        values = values[: len(values) // 2 * 2]
        timestamps = [
            (float(values[index]), float(values[index + 1]))
            for index in range(0, len(values), 2)
        ]
    return timestamps


def score_prediction(
    raw_output: str,
    ground_truth: Sequence[Sequence[float]],
    duration_s: float,
    metric: str,
) -> Dict[str, Any]:
    strict = parse_interval_set_strict(raw_output, duration_s=duration_s)
    strict_quality = set_tiou(strict.intervals, ground_truth) if strict.parse_valid else 0.0
    official_intervals = extract_time_official(raw_output)
    official_quality = (
        single_span_iou(official_intervals[0], ground_truth[0])
        if official_intervals and ground_truth
        else 0.0
    )
    official_parse = IntervalParse(
        raw_text="" if raw_output is None else str(raw_output),
        payload_text=None,
        payload_char_start=None,
        payload_char_end=None,
        intervals=[list(span) for span in official_intervals],
        parse_valid=bool(official_intervals),
        syntax_valid=bool(official_intervals),
        parse_error_code=(
            None
            if official_intervals
            else ("empty_output" if not str(raw_output or "").strip() else "other")
        ),
        parse_error_detail=(
            None
            if official_intervals
            else "official TimeLens regex found no timestamp pair"
        ),
    )
    selected_parse = (
        strict if metric == "strict_set_tiou" else official_parse
    )
    selected_quality = (
        strict_quality if metric == "strict_set_tiou" else official_quality
    )
    return {
        "parse": selected_parse.to_dict(),
        "strict_parse_valid": strict.parse_valid,
        "official_parse_valid": official_parse.parse_valid,
        "strict_set_tiou": strict_quality,
        "official_single_span_iou": official_quality,
        "quality": selected_quality,
        "success_iou_03": selected_quality >= 0.3,
        "success_iou_05": selected_quality >= 0.5,
        "success_iou_07": selected_quality >= 0.7,
    }
