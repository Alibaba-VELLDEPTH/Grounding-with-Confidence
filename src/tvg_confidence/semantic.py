from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .intervals import IntervalParse


@dataclass
class TokenSemantic:
    step_idx: int
    token_id: int
    token_text: str
    char_start: Optional[int]
    char_end: Optional[int]
    semantic_role: str
    semantic_event: Optional[str]
    span_index: Optional[int]
    value_piece_index: Optional[int]
    mapping_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _overlap(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def align_token_offsets(
    tokenizer: Any, token_ids: Sequence[int], decoded_text: str
) -> Tuple[List[Tuple[Optional[int], Optional[int]]], str]:
    ids = [int(token_id) for token_id in token_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    while ids and eos_id is not None and ids[-1] == eos_id:
        ids = ids[:-1]

    try:
        encoded = tokenizer(
            decoded_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if encoded_ids == ids and len(offsets) == len(ids):
            return offsets + [(len(decoded_text), len(decoded_text))] * (
                len(token_ids) - len(ids)
            ), "exact"
    except Exception:
        pass

    offsets: List[Tuple[Optional[int], Optional[int]]] = []
    previous = ""
    for step in range(len(ids)):
        current = tokenizer.decode(
            ids[: step + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not current.startswith(previous):
            return [(None, None)] * len(token_ids), "failed"
        offsets.append((len(previous), len(current)))
        previous = current

    if previous != decoded_text:
        return [(None, None)] * len(token_ids), "failed"
    offsets.extend([(len(decoded_text), len(decoded_text))] * (len(token_ids) - len(ids)))
    return offsets, "partial"


def _anchor_token_index(
    offsets: Sequence[Tuple[Optional[int], Optional[int]]],
    char_position: int,
    side: str,
) -> Optional[int]:
    valid = [
        (index, start, end)
        for index, (start, end) in enumerate(offsets)
        if start is not None and end is not None
    ]
    if not valid:
        return None
    if side == "before":
        before = [index for index, _, end in valid if end <= char_position]
        return max(before) if before else -1
    after = [
        index
        for index, start, end in valid
        if start < char_position <= end or end == char_position
    ]
    if after:
        return min(after)
    before = [index for index, _, end in valid if end <= char_position]
    return max(before) if before else -1


def map_token_semantics(
    tokenizer: Any,
    token_ids: Sequence[int],
    decoded_text: str,
    parsed: IntervalParse,
) -> Tuple[List[TokenSemantic], List[Dict[str, Any]], str]:
    offsets, alignment_status = align_token_offsets(tokenizer, token_ids, decoded_text)
    token_rows: List[TokenSemantic] = []
    anchors_by_token: Dict[int, List[str]] = {}
    anchor_rows: List[Dict[str, Any]] = []

    if parsed.syntax_valid:
        for anchor in parsed.semantic_anchors:
            token_index = _anchor_token_index(
                offsets, anchor.char_position, anchor.side
            )
            anchor_rows.append(
                {
                    "semantic_event": anchor.semantic_event,
                    "span_index": anchor.span_index,
                    "char_position": anchor.char_position,
                    "side": anchor.side,
                    "prefix_token_end": token_index,
                    "mapping_status": (
                        alignment_status if token_index is not None else "failed"
                    ),
                }
            )
            if token_index is not None:
                anchors_by_token.setdefault(token_index, []).append(
                    anchor.semantic_event
                )

    for step_idx, (token_id, offset) in enumerate(zip(token_ids, offsets)):
        start, end = offset
        role = "unparsed"
        span_index = None
        value_piece_index = None
        mapping_status = alignment_status
        if start is not None and end is not None:
            best_region = None
            best_overlap = 0
            for region in parsed.char_regions:
                overlap = _overlap((start, end), (region.char_start, region.char_end))
                if overlap > best_overlap:
                    best_region = region
                    best_overlap = overlap
            if best_region is not None:
                role = best_region.semantic_role
                span_index = best_region.span_index
                value_piece_index = best_region.value_piece_index
            elif parsed.payload_char_start is not None and end <= parsed.payload_char_start:
                role = "preamble"
            elif (
                parsed.payload_char_end is not None
                and start >= parsed.payload_char_end
            ):
                role = "eos"
            elif parsed.syntax_valid:
                role = "preamble"
        token_rows.append(
            TokenSemantic(
                step_idx=step_idx,
                token_id=int(token_id),
                token_text=tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                char_start=start,
                char_end=end,
                semantic_role=role,
                semantic_event=(
                    ",".join(sorted(set(anchors_by_token.get(step_idx, [])))) or None
                ),
                span_index=span_index,
                value_piece_index=value_piece_index,
                mapping_status=mapping_status,
            )
        )
    return token_rows, anchor_rows, alignment_status
