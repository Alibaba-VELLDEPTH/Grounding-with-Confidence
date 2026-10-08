"""Fixed TRAIN-only generation trends, independent of held-out model selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from .common import write_json
from .intervals import canonicalize_interval_set_to_duration, single_span_iou


def select_records(records, per_bucket=4):
    buckets = [[] for _ in range(4)]
    for row in records:
        n = len(row["gt_intervals"])
        bucket = 0 if n <= 1 else 1 if n <= 3 else 2 if n <= 7 else 3
        buckets[bucket].append(row)
    return [row for bucket in buckets for row in sorted(bucket, key=lambda r: hashlib.sha256(str(r["request_id"]).encode()).hexdigest())[:per_bucket]]


def generation_probe_callback(trainer, records, processor, collator_factory, make_messages, output_dir, config):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback, LogitsProcessorList
    from infer import CandidateAwareJsonProcessor
    from evaluation import set_metrics

    selected = select_records(records, int(config.get("per_gt_count_bucket", 4)))
    destination = Path(output_dir) / "train-generation-probe"
    collator = collator_factory(processor, "direct")

    class Probe(TrainerCallback):
        def run(self, state):
            started = time.monotonic()
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            model = trainer.accelerator.unwrap_model(trainer.model)
            generator = model.generator
            device = next(generator.parameters()).device
            device_ids = [device.index] if device.type == "cuda" else []
            training = model.training
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            rows = []
            try:
                with torch.random.fork_rng(devices=device_ids), torch.inference_mode():
                    model.eval()
                    for record in selected[rank::world_size]:
                        local = dict(record)
                        with collator._localize_video(record["video_path"]) as path:
                            local["video_path"] = path
                            messages = make_messages(local, record["direct_prompt"])
                            images, videos, info = collator._decode_video(messages)
                        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = collator._process_text(rendered, images, videos, info)
                        inputs = trainer._prepare_inputs(inputs)
                        prompt_length = int(inputs["input_ids"].shape[1])
                        budget = int(config["max_new_tokens"])
                        if prompt_length + budget > int(record["context_token_budget"]):
                            raise RuntimeError("train generation probe exceeds context budget")
                        stopper = CandidateAwareJsonProcessor(processor.tokenizer, prompt_length,
                            max_unique=32, duplicate_patience=3, max_total=40)
                        generated = generator.generate(**inputs, max_new_tokens=budget, do_sample=False,
                            use_cache=True, logits_processor=LogitsProcessorList([stopper]))
                        completion = generated[0, prompt_length:]
                        response = processor.tokenizer.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        parsed = canonicalize_interval_set_to_duration(response, record["source_duration_s"])
                        predictions = parsed.intervals if parsed.parse_valid else []
                        n = len(predictions)
                        duplicate = n - len({tuple(pair) for pair in predictions})
                        near_duplicates = sum(any(single_span_iou(pair, previous) >= .9 for previous in predictions[:i]) for i, pair in enumerate(predictions))
                        rows.append({"request_id": record["request_id"], "gt_count": len(record["gt_intervals"]),
                            "training_target_count": len(json.loads(record["direct_target"])),
                            "generated_count": n, "response": response,
                            "parse_valid": int(parsed.parse_valid), "completion_tokens": int(completion.numel()),
                            "token_budget_exhausted": int(completion.numel() >= budget), "stop_reason": stopper.stop_reason,
                            "exact_duplicate_fraction": duplicate / n if n else 0,
                            "near_duplicate_fraction_tiou09": near_duplicates / n if n else 0,
                            **set_metrics(predictions, record["gt_intervals"])})
                        del generated, inputs, images, videos, completion
            finally:
                model.train(training)
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
            gathered = [None] * world_size
            if dist.is_initialized():
                dist.all_gather_object(gathered, rows)
            else:
                gathered = [rows]
            if rank == 0:
                all_rows = [row for shard in gathered for row in shard]
                keys = ("gt_count", "training_target_count", "generated_count", "parse_valid", "completion_tokens",
                    "token_budget_exhausted", "exact_duplicate_fraction", "near_duplicate_fraction_tiou09",
                    "count_accuracy", "occurrence_precision_05", "occurrence_recall_05", "occurrence_f1_05", "miou", "set_tiou", "etf1")
                summary = {key: statistics.fmean(row[key] for row in all_rows) for key in keys}
                report = {"scope": "fixed stratified TRAIN samples; in-sample trend only; no threshold tuning",
                    "global_step": state.global_step, "queries": len(all_rows), "wall_s": time.monotonic() - started,
                    "metrics": summary, "rows": all_rows, "confidence_selection": False,
                    "metric_protocol": "existing set_metrics maximum-weight matching; miou uses official_multispan_iou; set_tiou uses merged interval unions"}
                write_json(destination / f"step-{state.global_step}.json", report)
                write_json(destination / "latest.json", report)
                print(json.dumps({"stage": "train_generation_probe", **{key: value for key, value in report.items() if key != "rows"}}, sort_keys=True), flush=True)
            # Probe time must not inflate the next training throughput window.
            trainer._diagnostic_window = (time.monotonic(), state.global_step)
            trainer._last_step_log_time = time.monotonic()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step == 0 and config.get("baseline", True):
                self.run(state)

        def on_save(self, args, state, control, **kwargs):
            self.run(state)

    return Probe()
