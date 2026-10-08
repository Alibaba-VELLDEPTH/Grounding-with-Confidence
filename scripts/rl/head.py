"""Span scoring and supervised online adaptation on detached old-policy hidden."""
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tvg_confidence.intervals import parse_interval_set_strict
from tvg_confidence.semantic import map_token_semantics
from reward import MAX_SOURCE_SPANS, pool
from tvg_confidence.intervals import single_span_iou

HEAD_SHA = os.environ.get("TVG_RL_HEAD_SHA", "")
ROLES = {"span_open", "start_value", "pair_delimiter", "end_value", "span_close"}


def validate_training_source(source, extra):
    if source == "omtg_rl":
        return
    experiment = json.loads(os.environ.get("TVG_RL_EXPERIMENT_IDENTITY", "{}"))
    phase = json.loads(os.environ.get("TVG_RL_DATA_PHASE", "{}"))
    phase_sha = hashlib.sha256(json.dumps(phase, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    index = extra.get("official_row_index")
    event_index = extra.get("official_event_index")
    if (source == "tl2_single_rl"
            and experiment.get("protocol") == "timelens2_v13_rl_tl2_single16k_v1"
            and experiment.get("data_arm") == phase.get("arm") == "tl2_single"
            and experiment.get("data_phase_sha256") == phase_sha
            and phase.get("source_counts") == {"omtg_rl": 8424, "tl2_single_rl": 7576}
            and phase.get("selected_rows") == 16000
            and phase.get("test_GT_used") is False
            and phase.get("heldout_video_overlap") == 0
            and isinstance(index, int) and index >= 0
            and isinstance(event_index, int) and event_index >= 0
            and extra.get("query_id") == f"tl2_single_rl_{index}_{event_index}"
            and extra.get("source_annotation_sha256") == phase.get("tl2_annotation_sha256")
            and bool(phase.get("tl2_annotation_sha256"))):
        return
    if (source == "omtg_sft_rl"
            and experiment.get("protocol") == "timelens2_v13_rl_difficulty16k_v1"
            and experiment.get("data_arm") == phase.get("arm")
            and phase.get("arm") in {"difficulty", "uniform"}
            and experiment.get("data_phase_sha256") == phase_sha
            and phase.get("selected_rows") == 16000
            and phase.get("test_GT_used") is False
            and phase.get("heldout_video_overlap") == 0
            and isinstance(index, int) and index >= 0
            and extra.get("query_id") == "omtg_sft_rl_" + str(index)):
        return
    raise RuntimeError("online head training must use authorized train rows in the frozen data phase")


class SpanScorer:
    def __init__(self, actor):
        path = Path(os.environ["TVG_RL_HEAD"])
        if len(HEAD_SHA) != 64:
            raise RuntimeError("Set TVG_RL_HEAD_SHA to the bootstrap head SHA-256")
        if hashlib.sha256(path.read_bytes()).hexdigest() != HEAD_SHA:
            raise RuntimeError("native v1.3 head identity mismatch")
        # Initialization/restoration must not consume the generator's RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(2560), torch.nn.Linear(2560, 256),
                torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1))
        self.head.load_state_dict(load_file(str(path)), strict=True)
        self.trainable = os.environ.get("TVG_RL_TRAIN_HEAD", "0") == "1"
        if self.trainable and os.environ["TVG_RL_ARM"] != "recall_head":
            raise RuntimeError("online head is authorized only for the recall_head arm")
        self.head.requires_grad_(self.trainable).eval()
        from online_head import OnlineHeadTrainer
        self.trainer = OnlineHeadTrainer(self.head) if self.trainable else None
        self.collect_training = False
        self.tokenizer = AutoTokenizer.from_pretrained(os.environ["TVG_RL_MODEL"], local_files_only=True)
        norms = [(name, module) for name, module in actor.named_modules()
                 if name.endswith("language_model.norm")]
        if len(norms) != 1:
            raise RuntimeError("expected exactly one final language-model norm: " + str([x[0] for x in norms]))
        self.norm = norms[0][1]
        self.hidden = None

    def capture(self, module, inputs, output):
        if self.hidden is not None:
            raise RuntimeError("multiple final-norm calls during one old-policy microbatch")
        self.hidden = output.detach()

    def score(self, inputs):
        hidden, self.hidden = self.hidden, None
        if hidden is None or hidden.ndim != 3 or hidden.shape[:2] != inputs["input_ids"].shape:
            raise RuntimeError("unpacked actor hidden/token alignment failed")
        response_width = inputs["responses"].shape[-1]
        start = hidden.shape[1] - response_width
        lengths = inputs["attention_mask"][:, start:].sum(-1).tolist()
        scores = torch.full((len(lengths), MAX_SOURCE_SPANS), -1.0, device=hidden.device, dtype=torch.float32)
        self.head.to(device=hidden.device, dtype=torch.float32)
        for i, length in enumerate(lengths):
            ids = inputs["responses"][i, :int(length)].tolist()
            text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parsed = parse_interval_set_strict(text)
            if not parsed.syntax_valid:
                continue
            if len(parsed.intervals) > MAX_SOURCE_SPANS:
                raise RuntimeError("source span score capacity exceeded")
            tokens, _, alignment = map_token_semantics(self.tokenizer, ids, text, parsed)
            if alignment not in {"exact", "partial"}:
                raise RuntimeError("span mapping failed: " + alignment)
            pooled_spans = []
            for j in range(len(parsed.intervals)):
                positions = [start + t.step_idx for t in tokens if t.span_index == j and t.semantic_role in ROLES]
                if not positions:
                    raise RuntimeError("empty source span token map")
                pooled = hidden[i, positions].float().mean(0, keepdim=True)
                pooled_spans.append(pooled)
                with torch.no_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
                    scores[i, j] = self.head(pooled).squeeze().sigmoid()
            if self.collect_training:
                extra = inputs["extra_info"][i]
                validate_training_source(inputs["data_source"][i], extra)
                gt = json.loads(inputs["reward_model"][i]["ground_truth"])
                if not gt or any(not 0 <= a < b <= extra["duration_s"] + 1e-4 for a, b in gt):
                    raise RuntimeError("invalid online-head training GT")
                canonical, _, survivors, indices = pool(text, extra["duration_s"])
                if canonical.parse_valid and survivors:
                    features = torch.cat([pooled_spans[j] for j in indices])
                    labels = [float(max(single_span_iou(span, target) for target in gt) >= 0.5)
                              for span in survivors]
                    self.trainer.add(features, labels)
        return scores


def scorer(actor):
    if actor.use_remove_padding or actor.use_fused_kernels or actor.use_ulysses_sp:
        raise RuntimeError("native head requires unpacked final-norm states")
    if not hasattr(actor, "tvg_scorer"):
        actor.tvg_scorer = SpanScorer(actor.actor_module)
    return actor.tvg_scorer


def start_batch(actor, data):
    current = scorer(actor)
    current.collect_training = current.trainable and data.meta_info.get("tvg_head_train", False)
    if current.collect_training:
        if data.meta_info.get("validate", False):
            raise RuntimeError("validation samples cannot update the confidence head")
        current.trainer.start_batch()


def begin(actor):
    current = scorer(actor)
    return current.norm.register_forward_hook(current.capture)


def update(actor):
    current = scorer(actor)
    if not current.trainable:
        return {}
    if not current.collect_training:
        raise RuntimeError("head update without a training old-policy pass")
    current.collect_training = False
    return current.trainer.update()


def save(actor, directory, step):
    current = scorer(actor)
    if current.trainable:
        current.trainer.save(Path(directory), step)


def restore(actor, directory):
    current = scorer(actor)
    if current.trainable:
        current.trainer.restore(Path(directory))
