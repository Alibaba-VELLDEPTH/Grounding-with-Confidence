"""Validate the pairing between the released generator and confidence head."""

PROTOCOL = "timelens2_v13_grpo_online_span_head_v1"
CONTRACT = {
    "version": "onpolicy_nms_iou05_binary_v1", "target": "max_GT_IoU_ge_0.5",
    "scope": "OMTG_RL_train_NMS_survivors", "hidden": "detached_old_policy_consumed_token_mean",
    "loss": "BCEWithLogits_candidate_mean_then_rollout_mean", "lr": 1e-4,
    "weight_decay": 0.01, "grad_clip": 1.0, "updates_per_outer_step": 1,
    "reward_order": "score_before_head_update;update_after_actor;next_batch_uses_updated_head",
    "dropout": 0.1, "rng_seed": 20260910,
}
SPAN_ROLES = ["span_open", "start_value", "pair_delimiter", "end_value", "span_close"]

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
