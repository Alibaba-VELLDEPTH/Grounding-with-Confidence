"""Compact, identity-bound Trainer checkpoints for full bounded-pool SFT."""
from __future__ import annotations

import hashlib
import json
import dataclasses
import errno
from pathlib import Path
import time

from .common import sha256_file, write_json


def required_checkpoint_files(world_size):
    files = {"model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"}
    files.update({"rng_state.pth"} if world_size == 1 else {f"rng_state_{rank}.pth" for rank in range(world_size)})
    return files


def visible_checkpoint_hash(path, expected_sha256=None, timeout_s=60.0):
    """Wait for a complete read; a just-closed shared file can still read empty."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            expected_size = path.stat().st_size
            digest, bytes_read = hashlib.sha256(), 0
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    bytes_read += len(block)
            actual = digest.hexdigest()
            if expected_size <= 0 or bytes_read != expected_size or path.stat().st_size != expected_size:
                reason = "empty, short or changing read"
            elif expected_sha256 is not None and actual != expected_sha256:
                reason = "persisted Trainer state differs from the state being saved"
            else:
                return actual
        except OSError as exc:
            if exc.errno not in (errno.EIO, errno.ENOENT, errno.EINTR):
                raise
            reason = f"temporary read error {exc.errno}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"checkpoint file not ready: {path.name}: {reason}")
        time.sleep(min(.5, remaining))


def checkpoint_file_hashes(root, world_size, state, timeout_s=60.0):
    """Bind the marker to actual bytes, including the exact live Trainer state."""
    paths = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file() and p.name != "_COMPLETE"}
    if not required_checkpoint_files(world_size) <= paths.keys():
        raise RuntimeError("refusing to publish an incomplete resumable checkpoint")
    # This is the same serialization as the pinned TrainerState.save_to_json.
    # Check the last-written metadata before hashing the larger model files.
    expected_state = (json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_sha = hashlib.sha256(expected_state).hexdigest()
    files = {"trainer_state.json": visible_checkpoint_hash(paths["trainer_state.json"], expected_sha, timeout_s)}
    for name, path in paths.items():
        if name != "trainer_state.json":
            files[name] = visible_checkpoint_hash(path, timeout_s=timeout_s)
    # Reject any later mutation before publishing the completion marker.
    visible_checkpoint_hash(paths["trainer_state.json"], expected_sha, timeout_s)
    return files


def checkpoint_contract(manifest, entry_sha256, world_size):
    fields = {"protocol_version": manifest["protocol_version"], "input_sha256": manifest["input"]["sha256"],
        "teacher_sha256": manifest["confidence_supervision"].get("teacher_sha256"),
        "base_model_revision": manifest["model"]["revision"], "generator_initialization": manifest.get("generator_initialization"),
        "training_contract": manifest["training_contract"], "head_initialization": manifest["head_initialization"],
        "entry_sha256": entry_sha256, "checkpoint_implementation_sha256": sha256_file(Path(__file__)), "world_size": world_size}
    if manifest["confidence_supervision"]["kind"] == "inline_gt_binary":
        fields["confidence_supervision"] = manifest["confidence_supervision"]
        fields["gt_supervision_implementation_sha256"] = sha256_file(Path(__file__).with_name("gt_binary_supervision.py"))
    return {**fields, "identity_sha256": hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def accept_checkpoint(path, contract):
    path = Path(path)
    marker = path / "_COMPLETE"
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text())
    if value["identity_sha256"] != contract["identity_sha256"]:
        raise RuntimeError("checkpoint data/model/training/implementation identity mismatch")
    required = required_checkpoint_files(contract["world_size"])
    if not required <= set(value["files"]):
        raise RuntimeError("checkpoint lacks trainable weights, optimizer, scheduler or per-rank RNG")
    for name, expected in value["files"].items():
        file = path / name
        if not file.is_file() or sha256_file(file) != expected:
            raise RuntimeError(f"checkpoint file identity mismatch: {name}")
    state = json.loads((path / "trainer_state.json").read_text())
    if state["global_step"] != value["global_step"]:
        raise RuntimeError("checkpoint optimizer-step identity mismatch")
    return True


def latest_checkpoint(root, contract):
    paths = sorted(Path(root).glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]), reverse=True)
    for path in paths:
        if accept_checkpoint(path, contract):
            return str(path)
    return None


class CompactCheckpointMixin:
    compact_checkpoint_contract = None

    def _save(self, output_dir=None, state_dict=None):
        if self.compact_checkpoint_contract is None:
            return super()._save(output_dir, state_dict)
        import torch
        from safetensors.torch import save_file

        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "_COMPLETE").exists():
            raise RuntimeError("refusing to overwrite a completed training checkpoint")
        model = self.accelerator.unwrap_model(self.model)
        trainable = {name: parameter.detach().cpu().contiguous().clone()
                     for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not any("confidence_head" in name for name in trainable) or not any("lora_" in name for name in trainable):
            raise RuntimeError("full joint checkpoint is missing trainable adapter/head parameters")
        save_file(trainable, str(destination / "model.safetensors"))
        torch.save(self.args, destination / "training_args.bin")
        adapter = destination / "generator-adapter"
        model.generator.save_pretrained(adapter, safe_serialization=True)
        head = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.confidence_head.state_dict().items()}
        save_file(head, str(destination / "span-confidence-head.safetensors"))
        write_json(destination / "span-head-config.json", {
            "schema_version": 1, "protocol_version": self.compact_checkpoint_contract["protocol_version"],
            "pooling": "mean over target interval text tokens",
            "target": self.compact_checkpoint_contract.get("confidence_supervision", {}).get("target", "verifier_v2_yes_probability"),
            "ranking_weight": self.compact_checkpoint_contract["training_contract"].get("confidence_ranking_weight", .25),
            "ranking_min_gap": .1, "generator_confidence_stop_gradient": True,
            "head_architecture": self.compact_checkpoint_contract["head_initialization"]["architecture"],
            "generator_identity": {"adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json")}})

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        result = super()._save_checkpoint(model, trial, *args, **kwargs)
        if self.compact_checkpoint_contract is not None:
            self.accelerator.wait_for_everyone()
            if self.is_world_process_zero():
                root = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
                files = checkpoint_file_hashes(root, self.compact_checkpoint_contract["world_size"], self.state)
                write_json(root / "_COMPLETE", {"identity_sha256": self.compact_checkpoint_contract["identity_sha256"],
                    "global_step": self.state.global_step, "files": files, "resumable": True})
                print(json.dumps({"stage": "resumable_checkpoint_complete", "global_step": self.state.global_step, "path": str(root)}), flush=True)
            self.accelerator.wait_for_everyone()
        return result

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self.compact_checkpoint_contract is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        from safetensors.torch import load_file

        if not accept_checkpoint(resume_from_checkpoint, self.compact_checkpoint_contract):
            raise RuntimeError("cannot resume a checkpoint without a verified completion marker")
        target = self.accelerator.unwrap_model(model if model is not None else self.model)
        trainable = {name for name, parameter in target.named_parameters() if parameter.requires_grad}
        weights = load_file(str(Path(resume_from_checkpoint) / "model.safetensors"), device="cpu")
        if set(weights) != trainable:
            raise RuntimeError("checkpoint trainable-parameter names differ from the fresh model")
        result = target.load_state_dict(weights, strict=False)
        if result.unexpected_keys or trainable & set(result.missing_keys):
            raise RuntimeError("checkpoint did not restore all trainable parameters")


def progress_callback(output_dir, save_final=False):
    from datetime import datetime, timezone
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if save_final and state.global_step == state.max_steps:
                control.should_save = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs and "loss" in logs:
                record = {
                    "updated_at": datetime.now(timezone.utc).isoformat(), "global_step": state.global_step,
                    "target_steps": state.max_steps, "epoch": state.epoch,
                    **{key: value for key, value in logs.items() if isinstance(value, (int, float))}}
                write_json(Path(output_dir) / "training-progress.json", record)
                with (Path(output_dir) / "training-metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if any(key.startswith("diag/") for key in logs):
                    write_json(Path(output_dir) / "training-diagnostics-latest.json", record)

    return Progress()
