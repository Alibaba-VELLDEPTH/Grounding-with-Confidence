#!/usr/bin/env python3
"""One-driver, eight-GPU Ray GRPO launcher. Dry-run unless --execute is given.

Checkpoint resume uses torch pickle: accept only trusted, self-produced files.
All inputs are local. Only a verified cache COPY of OMTG is ever patched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts/rl"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(PACKAGE)]
from actor_lr_phase import canonical_sha
from merge_sft import verify_merged
from patch_verl import REVISION, patched_sources, stage_upstream, verify_revision
from prepare_rl import load_prepared, validate_disjoint
from runtime import sha, verify_complete, write_json

PHASES = {
    "warmup": {"start": 0, "stop": 50, "save": 50, "first": 50, "lr": 1e-6, "reward": "recall_head"},
    "online": {"start": 50, "stop": 100, "save": 50, "first": 51, "lr": 1e-6, "reward": "recall_head"},
    "low-lr": {"start": 100, "stop": 200, "save": 25, "first": 101, "lr": 2e-7,
               "reward": "recall_batch_global_f1_tiou"},
}
RANK_KEYS = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
             "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "RAY_ADDRESS"}


def clean_environment(source):
    return {k: v for k, v in source.items() if k not in RANK_KEYS
            and not k.startswith(("TORCHELASTIC_", "TVG_RL_"))
            and k not in {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTHONPATH"}}


def build_environment(args, experiment, upstream_copy, inherited=None):
    env = clean_environment(os.environ if inherited is None else inherited)
    cache = args.cache_dir
    paths = [ROOT / "src", ROOT / "scripts", PACKAGE, upstream_copy / "rl/verl", upstream_copy / "rl"]
    env.update(PYTHONPATH=os.pathsep.join(map(str, paths)), TVG_RL_CACHE=str(cache),
        TVG_RL_OUTPUT=str(args.output_dir), TVG_RL_ARM="recall_head", TVG_RL_HEAD_SHA=args.head_sha256,
        TVG_RL_HEAD=str(args.head), TVG_RL_MODEL=str(args.model),
        TVG_RL_UPSTREAM_COPY=str(upstream_copy),
        TVG_RL_TRAIN_HEAD="0" if args.phase == "warmup" else "1",
        TVG_RL_ACTOR_PERFORMANCE="1" if args.phase == "low-lr" else "0",
        TVG_RL_FIRST_CHECKPOINT_STEP=str(PHASES[args.phase]["first"]),
        TVG_RL_EXPERIMENT_IDENTITY=json.dumps(experiment, sort_keys=True),
        TMPDIR=str(cache), HF_HOME=str(cache / "hf"), HF_DATASETS_CACHE=str(cache / "datasets"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="offline",
        PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
        VLLM_WORKER_MULTIPROC_METHOD="spawn", FORCE_QWENVL_VIDEO_READER="decord",
        OMP_NUM_THREADS="2", RAY_DEDUP_LOGS="0")
    if args.phase != "warmup":
        env.update(TVG_RL_HEAD_BOOTSTRAP_STEP="50", TVG_RL_HEAD_BOOTSTRAP_CHECKPOINT=str(args.bootstrap_checkpoint))
    if args.phase == "low-lr":
        env.update(TVG_RL_REWARD_VARIANT=PHASES[args.phase]["reward"],
                   TVG_RL_ACTOR_LR_PHASE=json.dumps(experiment["actor_lr_phase"], sort_keys=True))
    return env


def build_overrides(args):
    """Historical entry.py argv, with local paths and bounded STOP50/100/200."""
    phase = PHASES[args.phase]
    values = {
        "data.train_files": [str(args.train)], "data.val_files": [str(args.val)],
        "data.train_batch_size": 64, "data.val_batch_size": 32, "data.prompt_key": "prompt", "data.truncation": "error",
        "data.image_patch_size": 16, "data.max_prompt_length": 15872, "data.max_response_length": 512,
        "data.dataloader_num_workers": 1, "data.filter_overlong_prompts": False, "data.use_shm": False, "data.seed": 42,
        "actor_rollout_ref.model.path": str(args.model), "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.use_fused_kernels": False, "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.enable_activation_offload": True, "actor_rollout_ref.model.use_shm": False,
        "actor_rollout_ref.actor.strategy": "fsdp2", "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1, "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.use_torch_compile": False, "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.optim.lr": phase["lr"], "actor_rollout_ref.actor.optim.lr_warmup_steps": 5,
        "actor_rollout_ref.actor.optim.weight_decay": 0.1, "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2, "actor_rollout_ref.actor.clip_ratio_high": 0.285,
        "actor_rollout_ref.actor.clip_ratio_c": 10.0, "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean", "actor_rollout_ref.actor.entropy_from_logits_with_chunking": True,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True, "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.offload_policy": True,
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.n": 8, "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 0.8, "actor_rollout_ref.rollout.top_k": 20,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 2, "actor_rollout_ref.rollout.max_num_seqs": 64,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.65, "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True, "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 16384, "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1, "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False, "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.ref.strategy": "fsdp2", "actor_rollout_ref.nccl_timeout": 3600,
        "algorithm.adv_estimator": "grpo", "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.use_kl_in_reward": False, "algorithm.kl_ctrl.kl_coef": 0.0,
        "algorithm.rollout_correction.rollout_is": "token", "algorithm.rollout_correction.rollout_is_threshold": 2.0,
        "reward_model.reward_manager": "batch", "reward_model.launch_reward_fn_async": False,
        "custom_reward_function.path": str(PACKAGE / ("global_f1_reward.py" if args.phase == "low-lr" else "reward.py")),
        "custom_reward_function.name": "batch_reward",
        "trainer.logger": ["console"], "trainer.project_name": "tvg_confidence",
        "trainer.experiment_name": "public_rl_" + args.phase,
        "trainer.n_gpus_per_node": 8, "trainer.nnodes": 1, "trainer.total_epochs": 3,
        "trainer.total_training_steps": phase["stop"], "trainer.val_before_train": True, "trainer.test_freq": phase["save"],
        "trainer.save_freq": phase["save"], "trainer.max_actor_ckpt_to_keep": 3,
        "trainer.default_local_dir": str(args.output_dir), "trainer.balance_batch": False,
        "trainer.resume_mode": "resume_path" if args.resume else "disable",
        "trainer.validation_data_dir": str(args.output_dir / "validation"),
    }
    if args.phase == "low-lr":
        values["trainer.rollout_data_dir"] = str(args.output_dir / "rollouts")
    if args.resume:
        values["trainer.resume_from_path"] = str(args.resume)
    return values


def hydra_value(value):
    if isinstance(value, str):
        # Quoted Hydra scalar, not shell quoting. Suppress OmegaConf interpolation in paths.
        if "${" in value or "\n" in value or "\r" in value:
            raise ValueError("Hydra paths may not contain interpolation or newlines")
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(v) for v in value) + "]"
    return str(value).lower() if isinstance(value, bool) else str(value)


def command_for(overrides):
    return [sys.executable, "-u", "-m", "verl.trainer.main_ppo"] + [
        key + "=" + hydra_value(value) for key, value in overrides.items()]


def checkpoint(path, head_sha):
    """Fail early for absent files; full byte/hash verification never unpickles."""
    marker = Path(path) / "_COMPLETE.json"
    receipt = json.loads(marker.read_text())
    if receipt.get("source_head_sha256") != head_sha:
        raise ValueError("Checkpoint bootstrap head differs from --head-sha256")
    for name in receipt["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid checkpoint receipt path")
        file = Path(path) / name
        if file.is_symlink() or not file.is_file() or file.stat().st_size <= 0:
            raise ValueError("Missing/empty/symlinked checkpoint member: " + name)
    return verify_complete(path, require_online_head=bool(receipt.get("online_head")))


def validate_output(output, experiment, resume):
    if not output.exists():
        return
    plan = output / "workflow-plan.json"
    if not plan.is_file() or json.loads(plan.read_text()).get("experiment") != experiment:
        raise ValueError("Output exists and is not owned by this exact experiment")
    if resume is None:
        raise ValueError("Existing output requires an explicit complete --resume checkpoint")
    step = int(resume.name.removeprefix("global_step_"))
    for path in output.glob("global_step_*"):
        if not path.is_dir() or not (path / "_COMPLETE.json").is_file() or int(path.name.removeprefix("global_step_")) > step:
            raise ValueError("Output has a later or incomplete checkpoint; choose a fresh output directory")


def plan(args):
    if args.workers != 8:
        raise ValueError("This workflow requires exactly eight workers on one node")
    for key in ("model", "head", "train", "val", "upstream_root", "resume", "bootstrap_checkpoint"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve(strict=True))
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if len(args.head_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.head_sha256):
        raise ValueError("--head-sha256 must be an explicit lowercase SHA-256")
    if args.head.suffix != ".safetensors":
        raise ValueError("Bootstrap head must be a .safetensors file")
    for writable in (args.output_dir, args.cache_dir):
        for source in (args.upstream_root, args.model, args.resume, args.bootstrap_checkpoint):
            # An explicitly verified own-phase checkpoint may live inside its output.
            if writable == args.output_dir and source == args.resume and source and source.parent == writable:
                continue
            if source and (writable == source or source in writable.parents or writable in source.parents):
                raise ValueError("Output/cache and source directories must be separate")
    if args.output_dir == args.cache_dir or args.output_dir in args.cache_dir.parents or args.cache_dir in args.output_dir.parents:
        raise ValueError("Use separate output and cache directories")
    merged = verify_merged(args.model, args.head, args.head_sha256)
    verify_revision(args.upstream_root)
    patched_sources(args.upstream_root / "rl/verl")  # no mutation in dry-run
    train, val = load_prepared(args.train), load_prepared(args.val)
    videos = validate_disjoint(train, val)
    # Preserve the historical three epochs: do not silently stop before the target.
    if len(train) // 64 * 3 < PHASES[args.phase]["stop"]:
        raise ValueError("Train split is too small to reach this phase's target in three epochs")
    if len(val) % 8:
        raise ValueError("Monitor row count must be divisible by eight for static head scoring")
    implementation = set(PACKAGE.glob("*.py")) | set((ROOT / "src/tvg_confidence").glob("*.py"))
    implementation.update(ROOT / "scripts" / name for name in (
        "train_rl.py", "prepare_rl.py", "merge_sft.py", "video_budget.py"))
    lineage = {"merged_receipt_sha256": sha(args.model / "_MERGED.json"),
        "sft_source": merged["source_identity"], "train_sha256": sha(args.train), "val_sha256": sha(args.val),
        "video_content_sha256": {name: canonical_sha(files) for name, files in videos.items()},
        "upstream_revision": REVISION,
        "implementation": {str(p.relative_to(ROOT)): sha(p) for p in sorted(implementation)}}
    phase = PHASES[args.phase]
    experiment = {"protocol": "grounding_confidence_public_rl_v1", "phase": args.phase,
        "variant": phase["reward"], "start_step": phase["start"], "target_step": phase["stop"],
        "first_checkpoint_step": phase["first"], "lineage": lineage}
    resume = checkpoint(args.resume, args.head_sha256) if args.resume else None
    if args.phase != "warmup":
        if args.bootstrap_checkpoint is None or resume is None:
            raise ValueError("Online phases require --resume and --bootstrap-checkpoint global_step_50")
        bootstrap = checkpoint(args.bootstrap_checkpoint, args.head_sha256)
        if (bootstrap["step"] != 50 or bootstrap.get("online_head")
                or bootstrap.get("experiment", {}).get("phase") != "warmup"
                or bootstrap["experiment"].get("lineage") != lineage):
            raise ValueError("Bootstrap must be this lineage's complete frozen-head cp50")
        experiment["bootstrap_marker_sha256"] = sha(args.bootstrap_checkpoint / "_COMPLETE.json")
    elif args.bootstrap_checkpoint:
        raise ValueError("Warmup does not use --bootstrap-checkpoint")
    if resume:
        parent = resume["step"] == phase["start"] and args.phase != "warmup"
        prior = resume.get("experiment", {})
        if parent:
            expected_phase = "warmup" if args.phase == "online" else "online"
            if prior.get("phase") != expected_phase or prior.get("lineage") != lineage:
                raise ValueError("Wrong parent phase or model/data/code lineage")
            if bool(resume.get("online_head")) != (args.phase == "low-lr"):
                raise ValueError("Parent head state does not match phase transition")
            experiment["parent_marker_sha256"] = sha(args.resume / "_COMPLETE.json")
            if args.phase == "online" and args.resume != args.bootstrap_checkpoint:
                raise ValueError("Online phase must start from the explicit bootstrap cp50")
            if args.phase == "low-lr":
                if prior.get("bootstrap_marker_sha256") != experiment["bootstrap_marker_sha256"]:
                    raise ValueError("cp100 was trained from another cp50")
                experiment["actor_lr_phase"] = {"start_step": 100, "parent_actor_lr": 1e-6,
                    "actor_lr": 2e-7, "parent_marker_sha256": experiment["parent_marker_sha256"]}
        else:
            if not phase["start"] < resume["step"] < phase["stop"]:
                raise ValueError("Resume must be the phase parent or an unfinished own-phase checkpoint")
            for name in ("parent_marker_sha256", "actor_lr_phase"):
                if name in prior:
                    experiment[name] = prior[name]
        if args.phase == "low-lr":
            lr = experiment.get("actor_lr_phase", {})
            expected_lr = {"start_step": 100, "parent_actor_lr": 1e-6, "actor_lr": 2e-7,
                           "parent_marker_sha256": experiment.get("parent_marker_sha256")}
            if lr != expected_lr or not lr.get("parent_marker_sha256"):
                raise ValueError("Invalid original L actor-LR phase")
            experiment["actor_lr_phase_sha256"] = canonical_sha(lr)
        if not parent:
            if prior != experiment or resume.get("first_checkpoint_step") != phase["first"]:
                raise ValueError("Resume belongs to another experiment")
            if bool(resume.get("online_head")) != (args.phase != "warmup"):
                raise ValueError("Own checkpoint online-head state mismatch")
    validate_output(args.output_dir, experiment, args.resume)
    copy = args.cache_dir / ("omtg-" + REVISION[:12] + "-" + sha(PACKAGE / "patch_verl.py")[:12])
    overrides = build_overrides(args)
    return {"experiment": experiment, "overrides": overrides, "argv": command_for(overrides),
        "patched_upstream": str(copy), "train_rows": len(train), "monitor_rows": len(val)}, build_environment(args, experiment, copy)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=PHASES, required=True)
    for name in ("model", "head", "train", "val", "output-dir", "cache-dir", "upstream-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--head-sha256", required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--bootstrap-checkpoint", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--execute", action="store_true", help="Create local copy/output and actually run GPU training")
    return p


def main():
    args = parser().parse_args()
    # Reject torchrun fan-out instead of creating eight competing Ray drivers.
    if int(os.environ.get("RANK", "0")) != 0 or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        raise ValueError("Launch this once with python, not torchrun")
    for key in list(os.environ):
        if key.startswith("TVG_RL_"):
            del os.environ[key]
    configuration, env = plan(args)
    if not args.execute:
        print(json.dumps({"status": "dry_run", "gpu_executed": False, **configuration,
            "environment": {k: v for k, v in env.items() if k.startswith("TVG_RL_") or k == "PYTHONPATH"}}, indent=2))
        return 0
    env["TVG_RL_EXECUTE"] = "1"
    os.environ.clear()
    os.environ.update(env)
    import torch
    if torch.cuda.device_count() != 8:
        raise RuntimeError("Requires exactly eight visible CUDA GPUs; no host-brand restriction")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    upstream = stage_upstream(args.upstream_root, Path(configuration["patched_upstream"]))
    sys.path[:0] = env["PYTHONPATH"].split(os.pathsep)
    from runtime import verify_patched_imports
    verify_patched_imports()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created_output = True
    except FileExistsError:
        created_output = False
        validate_output(args.output_dir, configuration["experiment"], args.resume)
    import fcntl
    with (args.output_dir / ".driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not created_output:
            validate_output(args.output_dir, configuration["experiment"], args.resume)
        write_json(args.output_dir / "workflow-plan.json", configuration)
        command = configuration["argv"]
        probe = subprocess.run(command + ["--cfg", "job", "--resolve"], env=env,
            cwd=upstream / "rl/verl", text=True, capture_output=True)
        if probe.returncode:
            raise RuntimeError("Hydra/import preflight failed: " + probe.stderr[-6000:])
        (args.output_dir / "resolved-config.yaml").write_text(probe.stdout)
        result = subprocess.run(command, env=env, cwd=args.output_dir)
        write_json(args.output_dir / "driver-exit.json", {"exit_code": result.returncode})
        if result.returncode == 0:
            final = verify_complete(args.output_dir / f'global_step_{PHASES[args.phase]["stop"]}',
                require_online_head=args.phase != "warmup", experiment=configuration["experiment"])
            print(json.dumps({"status": "complete", "step": final["step"]}))
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
