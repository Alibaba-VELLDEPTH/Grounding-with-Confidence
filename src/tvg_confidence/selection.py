"""Generation-order NMS before confidence-based acceptance."""
from .intervals import canonicalize_interval_set_to_duration, single_span_iou

NMS = 0.3
MAX_CANDIDATES = 32

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
