"""Fail-closed patches to a PRIVATE COPY of pinned OMTG, never its checkout.

Upstream copyright/license headers are preserved in every copied source file.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

REVISION = "3291c9e19490db19c7aae593791f80aaca6f53a8"
COPY_MARKER = "_TVG_UPSTREAM_COPY.json"
PATCH_FILES = (
    "verl/models/transformers/monkey_patch.py", "verl/models/transformers/qwen3_vl.py",
    "verl/workers/actor/dp_actor.py", "verl/workers/fsdp_workers.py",
    "verl/workers/reward_manager/batch.py", "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/dataset/rl_dataset.py", "verl/utils/dataset/vision_utils.py",
    "verl/utils/tracking.py", "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    "verl/trainer/main_ppo.py", "verl/single_controller/ray/base.py",
)


def replace(root, file, old, new, count=1):
    text = root[file]
    if text.count(old) != count:
        raise ValueError(f"upstream patch anchor drift: {file}: {text.count(old)} != {count}")
    root[file] = text.replace(old, new)


def _apply(root):
    replace(root, "verl/models/transformers/monkey_patch.py",
        "        Qwen3VLModel.forward = qwen3_vl_base_forward\n",
        "        from qwen3_fsdp_compat import install_deepstack_clone\n"
        "        install_deepstack_clone()\n"
        "        Qwen3VLModel.forward = qwen3_vl_base_forward\n")
    actor = "verl/workers/actor/dp_actor.py"
    replace(root, actor, "        # set to eval\n        self.actor_module.eval()",
        "        from head import start_batch\n        start_batch(self, data)\n"
        "        # set to eval\n        self.actor_module.eval()")
    replace(root, actor,
        '        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]\n'
        '        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []',
        '        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]\n'
        '        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []\n'
        '        non_tensor_select_keys += ["extra_info", "reward_model", "data_source"]')
    replace(root, actor, "        log_probs_lst = []\n        entropy_lst = []",
        "        from head import begin\n        if use_dynamic_bsz:\n            raise RuntimeError('head scoring requires static microbatches')\n"
        "        tvg_scores = []\n        log_probs_lst = []\n        entropy_lst = []")
    old = """            with torch.no_grad():
                entropy, log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)"""
    new = """            hook = begin(self)
            try:
                with torch.no_grad():
                    entropy, log_probs, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    tvg_scores.append(self.tvg_scorer.score(model_inputs))
            finally:
                hook.remove()
            log_probs_lst.append(log_probs)"""
    replace(root, actor, old, new)
    replace(root, actor, "        return log_probs, entropys\n",
        "        self.tvg_span_scores = torch.cat(tvg_scores, dim=0)\n        return log_probs, entropys\n")
    replace(root, "verl/workers/fsdp_workers.py",
        'tensors={"old_log_probs": output, "entropys": entropys},',
        'tensors={"old_log_probs": output, "entropys": entropys, "tvg_span_scores": self.actor.tvg_span_scores},')
    replace(root, "verl/workers/fsdp_workers.py",
        "                metrics = self.actor.update_policy(data=data)\n",
        "                metrics = self.actor.update_policy(data=data)\n"
        "                from head import update as update_span_head\n"
        "                metrics.update(update_span_head(self.actor))\n")
    replace(root, "verl/workers/fsdp_workers.py",
        "        )\n        dist.barrier()\n\n        if self._is_lora and hasattr",
        "        )\n        from head import save as save_span_head\n"
        "        save_span_head(self.actor, local_path, global_step)\n"
        "        dist.barrier()\n\n        if self._is_lora and hasattr")
    replace(root, "verl/workers/fsdp_workers.py",
        "            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load\n"
        "        )\n\n        if self._is_offload_param:",
        "            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load\n"
        "        )\n        from head import restore as restore_span_head\n"
        "        restore_span_head(self.actor, local_path)\n"
        "        if os.environ.get('TVG_RL_ACTOR_LR_PHASE'):\n"
        "            from actor_lr_phase import restore as restore_actor_lr_phase\n"
        "            restore_actor_lr_phase(self.actor_optimizer, self.actor_lr_scheduler, local_path, self.rank)\n"
        "\n        if self._is_offload_param:")
    replace(root, "verl/workers/reward_manager/batch.py",
        '        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])',
        '        # DataProto.repeat preserves object references across rollouts.\n'
        '        # Per-response head scores must not mutate a shared prompt dictionary.\n'
        '        extras = [dict(item) for item in data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])]')
    replace(root, "verl/workers/reward_manager/batch.py",
        '            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]',
        '            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]\n'
        '            extras[i]["tvg_span_scores"] = data.batch["tvg_span_scores"][i].tolist()')
    trainer = "verl/trainer/ppo/ray_trainer.py"
    text = root[trainer]
    reward_anchor = '                    with marked_timer("reward", timing_raw, color="yellow"):\n'
    mode_anchor = '                    # Operating Mode Selection:'
    if text.count(reward_anchor) != 1 or text.count(mode_anchor) != 1:
        raise ValueError('reward-order patch anchor drift')
    start = text.index(reward_anchor)
    end = text.index(mode_anchor, start)
    reward = text[start:end]
    text = text[:start] + text[end:]
    anchor = '                    if self.use_reference_policy:\n'
    if text[start:].count(anchor) != 1:
        raise ValueError('old-policy/reward ordering anchor drift')
    where = text.index(anchor, start)
    root[trainer] = text[:where] + reward + text[where:]
    replace(root, "verl/trainer/ppo/ray_trainer.py",
        '                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)',
        '                            batch.meta_info["tvg_head_train"] = True\n'
        '                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)')
    replace(root, "verl/trainer/ppo/ray_trainer.py",
        '            # evaluate using reward_function\n',
        '            test_batch.meta_info["tvg_head_train"] = False\n'
        '            head_pass = self.actor_rollout_wg.compute_log_prob(test_batch)\n'
        '            test_batch.batch["tvg_span_scores"] = head_pass.batch["tvg_span_scores"]\n'
        '            # evaluate using reward_function\n')
    replace(root, "verl/trainer/ppo/ray_trainer.py",
        '        # latest checkpointed iteration tracker (for atomic usage)\n',
        '        from runtime import checkpoint_receipt\n'
        '        checkpoint_receipt(self.config.trainer.default_local_dir, self.global_steps)\n'
        '        # latest checkpointed iteration tracker (for atomic usage)\n')
    replace(root, "verl/trainer/ppo/ray_trainer.py",
        '                if self.config.trainer.save_freq > 0 and (\n'
        '                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration',
        '                from runtime import first_online_checkpoint\n'
        '                if self.config.trainer.save_freq > 0 and (\n'
        '                    first_online_checkpoint(self.global_steps) or is_last_step\n'
        '                    or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration')
    replace(root, "verl/utils/dataset/rl_dataset.py",
        '                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}',
        '                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False, "do_resize": False}', count=2)
    replace(root, "verl/utils/dataset/rl_dataset.py",
        '            input_ids = model_inputs.pop("input_ids")\n            attention_mask = model_inputs.pop("attention_mask")\n\n            if "second_per_grid_ts" in model_inputs:',
        '            from runtime import validate_video\n'
        '            validate_video(self.processor, videos, video_metadata, model_inputs, row_dict, row_dict_videos)\n'
        '            input_ids = model_inputs.pop("input_ids")\n            attention_mask = model_inputs.pop("attention_mask")\n\n            if "second_per_grid_ts" in model_inputs:')
    replace(root, "verl/utils/dataset/vision_utils.py", '    video = dict(video)\n',
        '    from runtime import cached_video\n    video = cached_video(dict(video))\n')
    replace(root, "verl/utils/tracking.py", '    def log(self, data, step, backend=None):\n',
        '    def log(self, data, step, backend=None):\n        from runtime import log_metrics\n        log_metrics(data, step)\n')
    rollout = "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    replace(root, rollout, '        compilation_config = {}\n',
        '        engine_kwargs["mm_processor_kwargs"] = {"do_sample_frames": False, "do_resize": False}\n'
        '        compilation_config = {}\n')
    replace(root, trainer, '                        if reward_extra_infos_dict:\n',
        '                        if reward_extra_infos_dict:\n'
        '                            if os.environ.get("TVG_RL_REWARD_VARIANT"):\n'
        '                                from global_f1_reward import training_diagnostics\n'
        '                                metrics.update(training_diagnostics(reward_extra_infos_dict, batch.non_tensor_batch.get("uid")))\n')
    # Assert provenance before Ray driver or workers can import installed/unpatched verl.
    main = 'verl/trainer/main_ppo.py'
    replace(root, main, 'from verl.experimental.dataset.sampler import AbstractSampler\n',
        'from runtime import verify_patched_imports, worker_environment\n'
        'verify_patched_imports()\n'
        'from verl.experimental.dataset.sampler import AbstractSampler\n')
    replace(root, main, '        default_runtime_env = get_ppo_ray_runtime_env()\n',
        '        default_runtime_env = get_ppo_ray_runtime_env()\n'
        '        default_runtime_env.setdefault("env_vars", {}).update(worker_environment())\n')
    replace(root, 'verl/single_controller/ray/base.py', '        if worker_env is not None:\n',
        '        from runtime import worker_environment\n'
        '        env_vars.update(worker_environment())\n'
        '        if worker_env is not None:\n')
    replace(root, 'verl/workers/fsdp_workers.py', 'import os\n',
        'import os\nfrom runtime import verify_patched_imports\nverify_patched_imports()\n')
    _performance(root)


def _performance(root):
    """Historical response-only 513-position projection and common left trim."""
    qwen = 'verl/models/transformers/qwen3_vl.py'
    text = root[qwen]
    start = text.index('def forward_with_normal_backend(')
    end = text.index('\ndef ', start + 4)
    normal = {qwen: text[start:end]}
    replace(normal, qwen, '    temperature: float = 1.0,\n',
        '    temperature: float = 1.0,\n    logits_to_keep: int = 0,\n')
    replace(normal, qwen, '    logits = self.lm_head(hidden_states)\n',
        '    if not isinstance(logits_to_keep, int) or not 0 <= logits_to_keep <= hidden_states.shape[1]:\n'
        '        raise ValueError("invalid response-only projection width")\n'
        '    if logits_to_keep:\n        hidden_states = hidden_states[:, -logits_to_keep:, :]\n'
        '    logits = self.lm_head(hidden_states)\n')
    root[qwen] = text[:start] + normal[qwen] + text[end:]
    actor = 'verl/workers/actor/dp_actor.py'
    replace(root, actor, '    def _forward_micro_batch(\n',
        '    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):\n'
        '        from actor_performance import forward_micro_batch\n'
        '        return forward_micro_batch(self, micro_batch, temperature, calculate_entropy)\n\n'
        '    def _forward_micro_batch_impl(\n')
    replace(root, actor, '        response_length = micro_batch["responses"].size(-1)\n',
        '        response_length = micro_batch["responses"].size(-1)\n'
        '        from actor_performance import trim_micro_batch, record_shape\n'
        '        optimized = getattr(self, "_tvg_optimized", True)\n'
        '        original = trimmed = micro_batch["input_ids"].shape[-1]\n'
        '        if optimized:\n            original, trimmed = trim_micro_batch(micro_batch)\n'
        '        record_shape(self, original, trimmed, response_length + 1 if optimized else original)\n')
    replace(root, actor, '            else:  # not using rmpad and no ulysses sp\n                extra_args = {}\n',
        '            else:  # not using rmpad and no ulysses sp\n'
        '                extra_args = {"logits_to_keep": response_length + 1} if optimized else {}\n')
    replace(root, actor, '        from head import start_batch\n',
        '        from actor_performance import reset_stats, shape_metrics\n'
        '        reset_stats(self)\n        from head import start_batch\n')
    replace(root, actor, '        return log_probs, entropys\n',
        '        self.tvg_old_shape_metrics = shape_metrics(self.tvg_shape_stats, "old_policy")\n'
        '        return log_probs, entropys\n')
    replace(root, actor, '        # make sure we are in training mode\n',
        '        from actor_performance import reset_stats, shape_metrics\n'
        '        reset_stats(self)\n        # make sure we are in training mode\n')
    replace(root, actor, '        return metrics\n',
        '        metrics.update({k: [v] for k, v in {**self.tvg_old_shape_metrics, **shape_metrics(self.tvg_shape_stats, "actor_update")}.items()})\n'
        '        return metrics\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patched_sources(root):
    """Validate all anchors and Python syntax in memory before writing anything."""
    root = Path(root)
    sources = {name: (root / name).read_text() for name in PATCH_FILES}
    _apply(sources)
    for name, text in sources.items():
        ast.parse(text, filename=name)
    return sources


def verify_revision(upstream):
    upstream = Path(upstream).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(upstream), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != REVISION:
        raise ValueError('OMTG revision must be ' + REVISION)
    if git('status', '--porcelain'):
        raise ValueError('OMTG source must be clean, including untracked files')
    if not (upstream / 'rl/verl/verl/trainer/main_ppo.py').is_file():
        raise ValueError('upstream-root must be the full OMTG checkout')
    return upstream


def copy_manifest(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != COPY_MARKER and '__pycache__' not in p.parts
            and p.suffix != '.pyc'}


def verify_copy(root):
    root = Path(root)
    receipt = json.loads((root / COPY_MARKER).read_text())
    if (receipt['revision'] != REVISION or receipt['patch_sha256'] != digest(__file__)
            or receipt['files'] != copy_manifest(root)):
        raise ValueError('cached patched OMTG source changed; choose a fresh cache')
    return root


def patch(root):
    """Idempotently verify a completed copy; unowned source trees are rejected."""
    root = Path(root).resolve(strict=True)
    return verify_copy(root.parents[1])


def source_files(upstream):
    """Copy tracked source only, not ignored credentials, virtualenvs or caches."""
    names = subprocess.check_output(['git', '--no-optional-locks', '-C', str(upstream),
                                     'ls-files', '-z'], text=True).split('\0')
    files = []
    for name in filter(None, names):
        relative = Path(name)
        path = upstream / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.is_file():
            raise ValueError('Unsupported tracked upstream source: ' + name)
        files.append(relative)
    return files


def stage_upstream(upstream, destination):
    """Copy locally, preserving upstream license files and never editing the checkout."""
    upstream = verify_revision(upstream)
    destination = Path(destination).resolve()
    if destination == upstream or upstream in destination.parents:
        raise ValueError('cache must be outside the upstream checkout')
    if destination.exists():
        return verify_copy(destination)
    files = source_files(upstream)
    # Validate against the original before making a potentially expensive copy.
    sources = patched_sources(upstream / 'rl/verl')
    source_identity = {str(name): digest(upstream / name) for name in files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.omtg-copy-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'source'
        staged.mkdir()
        for relative in files:
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(upstream / relative, path)
        verify_revision(upstream)  # Abort if tracked files changed during the copy.
        for name, text in sources.items():
            (staged / 'rl/verl' / name).write_text(text)
        receipt = {'revision': REVISION, 'patch_sha256': digest(__file__), 'files': copy_manifest(staged)}
        (staged / COPY_MARKER).write_text(json.dumps(receipt, sort_keys=True) + '\n')
        staged.rename(destination)
    return verify_copy(destination)
