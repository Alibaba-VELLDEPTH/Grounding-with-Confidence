"""Core GT-anchored target construction and offline confidence-replay attachment.

The functions are retained from the training-data builders. Dataset indexing,
cloud submission, fixed private input checksums, and job records are omitted.
PoolBuilder receives an argparse.Namespace with the settings in DATA_FORMAT.md.
"""
from collections import Counter
import hashlib
import json
import re
from tokenizers import Tokenizer

PROTOCOL = "timelens2_bounded_coverage_pool_sft_v1"
SUFFIX = "<|im_end|>\n"
PAIR = re.compile(r"\[\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*\]")

def stable_id(*values):
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()[:24]


def tiou(a, b):
    union = max(a[1], b[1]) - min(a[0], b[0])
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / union if union > 0 else 0.0


def render(slots):
    pieces, spans = ["["], []
    cursor = 1
    for i, slot in enumerate(slots):
        prefix = ",[" if i else "["
        pieces.append(prefix)
        cursor += len(prefix)
        coordinate_spans = []
        for j, value in enumerate(slot["interval"]):
            if j:
                pieces.append(",")
                cursor += 1
            text = json.dumps(float(value), allow_nan=False)
            pieces.append(text)
            coordinate_spans.append([cursor, cursor + len(text)])
            cursor += len(text)
        pieces.append("]")
        cursor += 1
        spans.append(coordinate_spans)
    pieces.append("]")
    return "".join(pieces), spans


class PoolBuilder:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenizer.from_file(str(args.tokenizer))

    def token_count(self, slots):
        return len(self.tokenizer.encode(render(slots)[0] + SUFFIX, add_special_tokens=False).ids)

    def anchors(self, gt):
        return [{"interval": g, "gt_index": i, "role": "exact_gt_anchor"}
                for i, g in enumerate(gt)]

    def valid_extra(self, interval, owner, gt, selected, owned_counts, minimum):
        return (interval[1] > interval[0]
                and owned_counts[owner] < self.args.extras_per_gt
                and tiou(interval, gt[owner]) >= minimum
                and all(tiou(interval, item["interval"]) < self.args.near_duplicate_tiou
                        for item in selected))

    def pool(self, unit, profile):
        gt, duration = unit["gt"], unit["meta"]["source_duration_s"]
        selected, owned_counts = self.anchors(gt), Counter()
        real = []
        for candidate in sorted(unit["proposals"]):
            if not (0 <= candidate[0] < candidate[1] <= duration):
                continue
            quality, owner = max((tiou(candidate, g), -i) for i, g in enumerate(gt))
            real.append((quality, -owner, candidate))
        # Prefer actual model alternatives; a round visits all occurrences
        # before allocating a second alternative to any occurrence.
        for round_index in range(self.args.extras_per_gt):
            for owner in range(len(gt)):
                options = sorted((x for x in real if x[1] == owner),
                                 key=lambda x: (-x[0], x[2]))
                for quality, _, candidate in options:
                    if not self.valid_extra(candidate, owner, gt, selected, owned_counts,
                                            self.args.min_model_tiou):
                        continue
                    slot = {"interval": list(candidate), "gt_index": owner,
                            "role": "retained_model_positive", "source_tiou": quality}
                    if len(selected) < self.args.max_candidates and self.token_count(selected + [slot]) <= self.args.target_tokens:
                        selected.append(slot)
                        owned_counts[owner] += 1
                    break
        if profile == "supported_expansion":
            # These are explicitly synthetic positive neighborhoods, not
            # unlabeled background fillers and not additional GT identities.
            patterns = ((-.2, -.2), (.2, .2), (-.2, .2), (.15, -.15),
                        (-.3, 0), (0, .3), (0, -.25), (.25, 0))
            for round_index in range(self.args.extras_per_gt):
                for owner, (start, end) in enumerate(gt):
                    width = end - start
                    options = set()
                    for left, right in patterns:
                        a = max(0.0, min(float(duration), round(start + width * left, 3)))
                        b = max(0.0, min(float(duration), round(end + width * right, 3)))
                        if self.valid_extra((a, b), owner, gt, selected, owned_counts,
                                            self.args.min_aug_tiou):
                            options.add((a, b))
                    ranked = sorted(options, key=lambda c: (
                        max(tiou(c, s["interval"]) for s in selected),
                        -tiou(c, gt[owner]), c))
                    for candidate in ranked:
                        slot = {"interval": list(candidate), "gt_index": owner,
                                "role": "synthetic_boundary_alternative",
                                "source_tiou": tiou(candidate, gt[owner])}
                        if len(selected) < self.args.max_candidates and self.token_count(selected + [slot]) <= self.args.target_tokens:
                            selected.append(slot)
                            owned_counts[owner] += 1
                            break
        return selected

    def row(self, uid, unit, profile):
        slots = self.pool(unit, profile)
        text, spans = render(slots)
        target = self.tokenizer.encode(text, add_special_tokens=False)
        suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        completion = self.tokenizer.encode(text + SUFFIX, add_special_tokens=False)
        if target.ids + suffix.ids != completion.ids:
            raise ValueError("target/suffix tokenizer boundary changed")
        counts = Counter()
        token_owners = []
        for start, end in target.offsets:
            owners = [(i, span) for i, pair in enumerate(spans) for span in pair
                      if max(start, span[0]) < min(end, span[1])]
            if owners:
                if len(owners) != 1 or not owners[0][1][0] <= start < end <= owners[0][1][1]:
                    raise ValueError("coordinate token crosses a loss-group boundary")
                group = "gt" if slots[owners[0][0]]["role"] == "exact_gt_anchor" else "alternative"
            else:
                group = "format"
            token_owners.append(group)
            counts[group] += 1
        counts["format"] += len(suffix.ids)
        # Existing trainer computes weighted token mean. Set coordinate
        # weights so its group masses are .6/.3/.1 (or .9/0/.1 without extras),
        # while grammar AND the external assistant suffix retain weight 1.
        format_mass = counts["format"]
        weights = {"gt": format_mass * (6 if counts["alternative"] else 9) / counts["gt"]}
        if counts["alternative"]:
            weights["alternative"] = 3 * format_mass / counts["alternative"]
        weighted = []
        for i, pair in enumerate(spans):
            group = "gt" if slots[i]["role"] == "exact_gt_anchor" else "alternative"
            for span in pair:
                weighted.append({"char_span": span, "weight": weights[group], "kind": group})
        prompt = (
            "You are given a video.\nTask: temporal retrieval candidate generation.\n"
            f'Find time spans in which the query "{unit["meta"]["query"]}" occurs.\n'
            "Build a high-recall candidate pool for a later confidence selector. "
            "First cover distinct occurrences throughout the video. Then include a few "
            "plausible alternative boundaries when useful. Overlapping alternatives are allowed, "
            "but avoid near-identical intervals and never repeat identical coordinates. "
            f"Return at most {self.args.max_candidates} candidates in seconds. Do not fill the "
            "budget with unrelated intervals. Close the array after the useful candidates.\n"
            "Output only one JSON array: [[start, end], ...]."
        )
        row = dict(unit["meta"])
        row.update({
            "schema_version": 1, "protocol_version": PROTOCOL,
            "request_id": PROTOCOL + "_" + stable_id(profile, uid),
            "coverage_unit_id": uid, "source_input_request_ids": sorted(unit["source_ids"]),
            "data_profile": profile, "gt_intervals": unit["gt"], "gt_cardinality": len(unit["gt"]),
            "original_gt_intervals": unit["original_gt"],
            "gt_duration_alignment": "explicit clipping only within declared rounding tolerance; originals retained",
            "prompt_text": prompt, "direct_prompt_text": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "direct_target_override": text, "direct_lm_masked_char_spans": [],
            "direct_lm_weighted_char_spans": weighted,
            "confidence_supervision_available": False, "confidence_backprop_to_generator": False,
            "confidence_supervision_definition": "none; head training requires real deployment rollouts",
            "completion_token_budget": self.args.decode_tokens,
            "context_token_budget": self.args.context_tokens,
            "expected_completion_tokens": len(completion.ids),
            "expected_completion_sha256": hashlib.sha256(json.dumps(completion.ids, separators=(",", ":")).encode()).hexdigest(),
            "pool_contract": {
                "candidate_cap": self.args.max_candidates, "gt_anchors": len(unit["gt"]),
                "target_candidates": len(slots), "max_alternatives_per_gt": self.args.extras_per_gt,
                "gt_is_always_retained": True, "masked_coordinate_fillers": 0,
                "close_after": "complete bounded candidate target, not the last GT anchor",
                "token_counts": dict(counts), "numeric_token_weights": weights,
                "synthetic_alternatives_are_not_new_occurrences": True,
            },
            "direct_candidate_slots": [{
                "candidate_id": stable_id(uid, s["role"], s["interval"]),
                "slot_role": s["role"], "assigned_gt_index": s["gt_index"],
                "assigned_gt_tiou": tiou(s["interval"], unit["gt"][s["gt_index"]]),
                "lm_interval_supervised": True, "end_char_span": spans[i][1],
            } for i, s in enumerate(slots)],
            "paper_main_eligible": False,
        })
        # Data-contract validation, performed on every materialized row.
        parsed = json.loads(text)
        assert parsed[:len(unit["gt"])] == unit["gt"]
        assert len(parsed) == len({tuple(x) for x in parsed}) <= self.args.max_candidates
        assert len(completion.ids) <= self.args.target_tokens < self.args.decode_tokens
        assert all(0 <= a < b <= unit["meta"]["source_duration_s"] for a, b in parsed)
        for i in range(len(unit["gt"]), len(slots)):
            s = slots[i]
            minimum = self.args.min_model_tiou if s["role"] == "retained_model_positive" else self.args.min_aug_tiou
            assert tiou(s["interval"], unit["gt"][s["gt_index"]]) >= minimum
            assert all(tiou(s["interval"], prior["interval"]) < self.args.near_duplicate_tiou for prior in slots[:i])
        assert max(Counter(s["gt_index"] for s in slots[len(unit["gt"]):]).values(), default=0) <= self.args.extras_per_gt
        return row


def attach_replay(row, source, tokenizer, used_teachers, roles):
    assert row["source_video_uid"] == source["source_video_uid"] and row["video_sha256"] == source["video_sha256"]
    assert row["query"] == source["query"]
    assert row["original_gt_intervals"] == sorted(source["gt_intervals"])
    replay = source["direct_target_override"]
    pairs = list(PAIR.finditer(replay))
    assert len(pairs) == len(source["direct_candidate_slots"]) == len(json.loads(replay))
    slots = []
    for match, slot in zip(pairs, source["direct_candidate_slots"], strict=True):
        cid = slot["candidate_id"]
        previous = used_teachers.setdefault(cid, row["source_video_uid"])
        assert previous == row["source_video_uid"]
        slots.append({"candidate_id": cid, "span_char_span": [match.start(), match.end()], "source_slot_role": slot["slot_role"]})
        roles[slot["slot_role"]] += 1
    ids = tokenizer.encode(replay + SUFFIX, add_special_tokens=False).ids
    assert len(ids) <= 2048
    row.update(confidence_supervision_available=True, confidence_backprop_to_generator=False,
        confidence_supervision_definition="unchanged verifier-v2 continuous P(Yes); soft BCE + ranking; detached span mean; replay LM=0",
        detached_confidence_replay_available=True, detached_confidence_replay_target=replay,
        detached_confidence_replay_prompt_text=source.get("direct_prompt_text") or source["prompt_text"],
        detached_confidence_replay_candidate_slots=slots, confidence_replay_source_request_id=source["request_id"],
        confidence_replay_distribution="historical offline mixed proposals; unchanged teacher scores; not v1.2 on-policy rollouts",
        replay_completion_token_budget=2048, expected_replay_tokens=len(ids),
        expected_replay_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest())
