"""One globally normalized binary head update per completed GRPO rollout batch."""
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file

CONTRACT = {
    "version": "onpolicy_nms_iou05_binary_v1", "target": "max_GT_IoU_ge_0.5",
    "scope": "OMTG_RL_train_NMS_survivors", "hidden": "detached_old_policy_consumed_token_mean",
    "loss": "BCEWithLogits_candidate_mean_then_rollout_mean", "lr": 1e-4,
    "weight_decay": 0.01, "grad_clip": 1.0, "updates_per_outer_step": 1,
    "reward_order": "score_before_head_update;update_after_actor;next_batch_uses_updated_head",
    "dropout": 0.1, "rng_seed": 20260910,
}


def distributed():
    return dist.is_available() and dist.is_initialized()


def summed(value):
    if distributed():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def state_digest(head, optimizer, updates):
    digest = hashlib.sha256(str(updates).encode())
    def add(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode())
    add(head.state_dict())
    add(optimizer.state_dict())
    return digest.hexdigest()


class OnlineHeadTrainer:
    def __init__(self, head):
        self.head = head
        self.optimizer = None
        self.updates = 0
        self.bootstrap_step = int(os.environ["TVG_RL_HEAD_BOOTSTRAP_STEP"])
        self.pending = []
        self.batch_open = False

    def ensure_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=CONTRACT["lr"],
                weight_decay=CONTRACT["weight_decay"])

    def start_batch(self):
        if self.batch_open or self.pending:
            raise RuntimeError("unconsumed head training batch")
        self.batch_open = True

    def add(self, features, labels):
        if not self.batch_open or len(features) != len(labels) or not labels:
            raise RuntimeError("invalid pending head batch")
        if features.requires_grad or features.grad_fn is not None or not torch.isfinite(features).all():
            raise RuntimeError("head features must be detached and finite")
        self.pending.append((features.detach().float().cpu(), torch.tensor(labels, dtype=torch.float32)))

    def update(self):
        if not self.batch_open:
            raise RuntimeError("no captured training batch")
        self.batch_open = False
        self.ensure_optimizer()
        device = next(self.head.parameters()).device
        count = summed(torch.tensor(float(len(self.pending)), device=device))
        if count.item() == 0:
            raise RuntimeError("no valid NMS candidates in the global head training batch")
        features = torch.cat([x for x, _ in self.pending]).to(device) if self.pending else torch.empty((0, 2560), device=device)
        labels = torch.cat([y for _, y in self.pending]).to(device) if self.pending else torch.empty(0, device=device)
        weights = torch.cat([torch.full((len(y),), 1.0 / len(y)) for _, y in self.pending]).to(device) if self.pending else labels
        self.pending.clear()
        self.optimizer.zero_grad(set_to_none=True)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(features).squeeze(-1)
            scores = logits.sigmoid()
            # Candidate totals are reduced independently of query-normalized training loss.
            stats = torch.stack([F.binary_cross_entropy_with_logits(logits, labels, reduction="sum"),
                ((scores - labels) ** 2).sum(), labels.sum(), labels.new_tensor(len(labels)),
                (scores * labels).sum(), (scores * (1 - labels)).sum()])
            summed(stats)
        self.head.train()
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        rank = dist.get_rank() if distributed() else 0
        # Dropout has its own deterministic stream; scoring remains in eval mode.
        with torch.random.fork_rng(devices=devices):
            seed = CONTRACT["rng_seed"] + self.updates * 1024 + rank
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
                logits = self.head(features).squeeze(-1)
                loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).sum() / count
                loss.backward()
        for parameter in self.head.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            summed(parameter.grad)
        loss_total = summed(loss.detach().clone())
        norm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), CONTRACT["grad_clip"])
        if not torch.isfinite(norm) or not torch.isfinite(loss_total) or not torch.isfinite(stats).all():
            raise RuntimeError("nonfinite online-head loss/gradient")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.head.eval()
        self.updates += 1
        bce, brier, positives, candidates, positive_score, negative_score = stats.tolist()
        return {"head/train_bce_before": bce / candidates, "head/train_brier_before": brier / candidates,
            "head/train_loss": loss_total.item(), "head/positive_rate": positives / candidates,
            "head/positive_score_mean": positive_score / max(1, positives),
            "head/negative_score_mean": negative_score / max(1, candidates - positives),
            "head/candidates": candidates, "head/valid_rollouts": count.item(),
            "head/grad_norm": norm.item(), "head/lr": CONTRACT["lr"], "head/updates": self.updates}

    def save(self, directory, step):
        from runtime import sha, write_json
        if self.pending or self.batch_open or step != self.bootstrap_step + self.updates:
            raise RuntimeError("head checkpoint must match the completed actor step")
        self.ensure_optimizer()
        digest = state_digest(self.head, self.optimizer, self.updates)
        replicas = [None] * dist.get_world_size() if distributed() else [digest]
        if distributed():
            dist.all_gather_object(replicas, digest)
        if len(set(replicas)) != 1:
            raise RuntimeError("head model/optimizer state diverged across actor ranks")
        if not distributed() or dist.get_rank() == 0:
            state = {"contract": CONTRACT, "global_step": step, "bootstrap_step": self.bootstrap_step,
                "updates": self.updates, "head": self.head.state_dict(), "optimizer": self.optimizer.state_dict(),
                "state_sha256": digest, "source_head_sha256": os.environ["TVG_RL_HEAD_SHA"]}
            path = directory / "span-head-training-state.pt"
            torch.save(state, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)
            weights = directory / "span-confidence-head.safetensors"
            save_file({k: v.detach().float().cpu().contiguous() for k, v in self.head.state_dict().items()}, str(weights) + ".tmp")
            os.replace(str(weights) + ".tmp", weights)
            write_json(directory / "span-head-replica-receipt.json", {"global_step": step,
                "updates": self.updates, "replica_state_sha256": replicas,
                "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in (path, weights)}})
        if distributed():
            dist.barrier()

    def restore(self, directory):
        from runtime import write_json
        step = int(directory.parent.name.removeprefix("global_step_"))
        path = directory / "span-head-training-state.pt"
        self.head.to(device=torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu", dtype=torch.float32)
        self.ensure_optimizer()
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if (state["contract"] != CONTRACT or state["global_step"] != step
                    or state["bootstrap_step"] != self.bootstrap_step
                    or state["updates"] != step - self.bootstrap_step
                    or state["source_head_sha256"] != os.environ["TVG_RL_HEAD_SHA"]):
                raise RuntimeError("online-head resume contract mismatch")
            self.head.load_state_dict(state["head"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.updates = state["updates"]
            if state_digest(self.head, self.optimizer, self.updates) != state["state_sha256"]:
                raise RuntimeError("restored head model/optimizer checksum mismatch")
            kind = "complete_online_head_resume"
        else:
            expected = Path(os.environ["TVG_RL_HEAD_BOOTSTRAP_CHECKPOINT"]) / "actor"
            if directory != expected or step != self.bootstrap_step or self.updates:
                raise RuntimeError("missing corresponding online head checkpoint")
            kind = "explicit_frozen_head_to_online_transition"
        self.head.eval()
        if not distributed() or dist.get_rank() == 0:
            write_json(Path(os.environ["TVG_RL_OUTPUT"]) / "head-resume-acceptance.json", {
                "status": "passed", "kind": kind, "actor_step": step, "head_updates": self.updates,
                "checkpoint": str(directory), "contract": CONTRACT})
