"""Export the online head paired with an immutable GRPO actor checkpoint.

Only trusted, self-produced checkpoints: CPU torch.load(weights_only=False)
uses pickle and can execute code. Hash verification is not a security sandbox.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

PROTOCOL = "timelens2_v13_grpo_online_span_head_v1"
CONTRACT = {
    "version": "onpolicy_nms_iou05_binary_v1", "target": "max_GT_IoU_ge_0.5",
    "scope": "OMTG_RL_train_NMS_survivors", "hidden": "detached_old_policy_consumed_token_mean",
    "loss": "BCEWithLogits_candidate_mean_then_rollout_mean", "lr": 1e-4,
    "weight_decay": 0.01, "grad_clip": 1.0, "updates_per_outer_step": 1,
    "reward_order": "score_before_head_update;update_after_actor;next_batch_uses_updated_head",
    "dropout": 0.1, "rng_seed": 20260910,
}
HEAD_FILES = (
    "span-confidence-head.safetensors", "span-head-training-state.pt",
    "span-head-replica-receipt.json",
)
SPAN_ROLES = ["span_open", "start_value", "pair_delimiter", "end_value", "span_close"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def training_state_digest(state):
    import torch
    digest = hashlib.sha256(str(state["updates"]).encode())

    def add(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            assert torch.isfinite(tensor).all().item(), "nonfinite head/optimizer tensor"
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

    add(state["head"])
    add(state["optimizer"])
    return digest.hexdigest()


def validate_config(config, head_sha256, full_model_manifest):
    """Require the actor and head from the same completed training step."""
    identity = config.get("generator_identity") or {}
    model = full_model_manifest or {}
    online = model.get("online_head") or {}
    checks = {
        "protocol": config.get("protocol_version") == PROTOCOL,
        "contract": config.get("training_contract") == CONTRACT,
        "target": config.get("target") == CONTRACT["target"],
        "pooling": config.get("pooling") == "mean over exact semantic consumed interval tokens",
        "semantic_roles": set(config.get("semantic_roles") or []) == set(SPAN_ROLES),
        "architecture": config.get("head_architecture") == {
            "input_dim": 2560, "hidden_dim": 256, "output_dim": 1, "dropout": 0.1},
        "hidden_gradient": config.get("generator_confidence_stop_gradient") is True,
        "generator_training": config.get("generator_frozen_during_training") is False,
        "full_model": model.get("status") == "complete" and model.get("adapter_loaded") is False,
        "checkpoint_marker": bool(identity.get("checkpoint_marker_sha256"))
            and identity.get("checkpoint_marker_sha256") == model.get("checkpoint_marker_sha256"),
        "checkpoint_step": identity.get("step") == model.get("step"),
        "head_steps": isinstance(config.get("head_updates"), int)
            and config["head_updates"] > 0
            and config["head_updates"] + config.get("bootstrap_step", -1) == model.get("step")
            and config["head_updates"] == online.get("updates"),
        "head_identity": config.get("head_sha256") == head_sha256 == online.get("head_sha256"),
    }
    if not all(checks.values()):
        raise ValueError(f"online head / actor pairing failed: {checks}")
    return checks


def export_head(source, marker_sha256, expected_step, output, *, trusted=False):
    if not trusted:
        raise ValueError('Explicit trust is required before deserializing torch pickle')
    if not __debug__:
        raise RuntimeError('Do not use Python -O: integrity assertions are required')
    import torch
    from safetensors.torch import load_file
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    marker = source / "_COMPLETE.json"
    assert sha(marker) == marker_sha256
    complete = json.loads(marker.read_text())
    assert complete["step"] == expected_step and complete["world_size"] == 8
    assert complete.get("online_head") is True, "checkpoint has no trained online head"
    assert not output.exists(), "inspect existing head export instead of overwriting"
    output.mkdir(parents=True)
    verified = {}
    for name in HEAD_FILES:
        identity = complete["files"]["actor/" + name]
        raw = (source / "actor" / name).read_bytes()
        assert len(raw) == identity["bytes"] and hashlib.sha256(raw).hexdigest() == identity["sha256"], name
        dest = output / name
        with dest.open("xb") as f:
            f.write(raw)
        assert sha(dest) == identity["sha256"], name
        verified[name] = identity
    state = torch.load(output / "span-head-training-state.pt", map_location="cpu", weights_only=False)
    replicas = json.loads((output / "span-head-replica-receipt.json").read_text())
    assert state["contract"] == CONTRACT
    assert state["global_step"] == replicas["global_step"] == expected_step
    experiment = complete.get("experiment", {})
    fresh_variants = {
        "timelens2_v13_recall_full300_v1": "recall_original",
        "timelens2_v13_l_full300_v1": "recall_batch_global_f1_tiou",
    }
    fresh = experiment.get("protocol") in fresh_variants
    if fresh:
        assert experiment.get("variant") == fresh_variants[experiment["protocol"]]
        assert experiment.get("start_step") == 0
        assert experiment.get("target_step") == 300
    bootstrap_step = 0 if fresh else 50
    assert state["bootstrap_step"] == bootstrap_step
    assert state["updates"] == replicas["updates"] == expected_step - bootstrap_step > 0
    assert state["source_head_sha256"] == complete["source_head_sha256"]
    digest = training_state_digest(state)
    assert digest == state["state_sha256"]
    assert replicas["replica_state_sha256"] == [digest] * 8
    assert all(replicas["files"][name] == verified[name] for name in HEAD_FILES[:2])
    weights = load_file(str(output / "span-confidence-head.safetensors"), device="cpu")
    assert set(weights) == set(state["head"])
    assert all(v.dtype == torch.float32 and torch.equal(v, state["head"][k]) for k, v in weights.items())
    head = torch.nn.Sequential(torch.nn.LayerNorm(2560), torch.nn.Linear(2560, 256),
        torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1))
    head.load_state_dict(weights, strict=True)
    adam = state["optimizer"]
    assert len(adam["state"]) == len(list(head.parameters())) == 6
    assert all(float(s["step"]) == state["updates"] for s in adam["state"].values())
    assert len(adam["param_groups"]) == 1
    assert adam["param_groups"][0]["lr"] == CONTRACT["lr"]
    assert adam["param_groups"][0]["weight_decay"] == CONTRACT["weight_decay"]
    config = {
        "protocol_version": PROTOCOL, "training_contract": CONTRACT,
        "target": CONTRACT["target"], "pooling": "mean over exact semantic consumed interval tokens",
        "semantic_roles": SPAN_ROLES, "hidden_layer": "final decoder layer",
        "head_architecture": {"input_dim": 2560, "hidden_dim": 256, "output_dim": 1, "dropout": 0.1},
        "generator_confidence_stop_gradient": True, "generator_frozen_during_training": False,
        "generator_identity": {"step": expected_step,
                               "checkpoint_marker_sha256": marker_sha256},
        "head_updates": state["updates"], "bootstrap_step": state["bootstrap_step"],
        "head_sha256": verified[HEAD_FILES[0]]["sha256"],
        "bootstrap_head_sha256": state["source_head_sha256"],
        "training_state_sha256": verified[HEAD_FILES[1]]["sha256"],
        "training_replica_state_digest": digest,
    }
    dump(output / "span-head-config.json", config)
    receipt = {
        "status": "complete", "scope": "online head artifacts only; actor export required separately",
        "checkpoint": str(source), "step": expected_step, "checkpoint_marker_sha256": marker_sha256,
        "head_path": str(output), "head_sha256": config["head_sha256"],
        "head_config_sha256": sha(output / "span-head-config.json"), "updates": state["updates"],
        "bootstrap_step": state["bootstrap_step"],
        "bootstrap_head_sha256": state["source_head_sha256"], "source_files": verified,
        "replica_state_digest": digest, "optimizer_updates_during_export": 0,
        "exporter_sha256": sha(Path(__file__)), "time": time.time(),
    }
    assert sha(marker) == marker_sha256, "checkpoint marker changed during head export"
    dump(output / "export-receipt.json", receipt)
    dump(output / "_HEAD_SUCCESS", {"export_receipt_sha256": sha(output / "export-receipt.json")})
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-marker-sha256", required=True)
    p.add_argument("--expected-step", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trust-self-produced-checkpoint", action="store_true", required=True)
    a = p.parse_args()
    r = export_head(a.checkpoint, a.expected_marker_sha256, a.expected_step, a.output,
                    trusted=a.trust_self_produced_checkpoint)
    print(json.dumps({k: r[k] for k in ("status", "scope", "step", "updates", "head_path", "head_sha256", "head_config_sha256")}))


if __name__ == "__main__":
    main()
