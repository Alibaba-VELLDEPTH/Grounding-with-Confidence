"""CPU-only workflow tests. Set TVG_TEST_OMTG_ROOT for real pinned patch tests.

No model loading, network, Ray cluster, GPU, or optimizer training is performed.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/rl"), str(ROOT / "src")]
import actor_lr_phase
import checkpoint_io
import export_rl
import merge_sft
import patch_verl
import prepare_rl
import runtime
import train_rl


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def marker(root, step, head="a" * 64, online=False, experiment=None):
    directory = root / f"global_step_{step}"
    names = ["data.pt"] + [f"actor/{kind}_world_size_8_rank_{rank}.pt"
        for rank in range(8) for kind in ("model", "optim", "extra_state")]
    if online:
        names += ["actor/" + name for name in (
            "span-confidence-head.safetensors", "span-head-training-state.pt", "span-head-replica-receipt.json")]
    files = {}
    for name in names:
        file = directory / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(name.encode())
        files[name] = {"bytes": file.stat().st_size, "sha256": prepare_rl.sha(file)}
    receipt = {"step": step, "world_size": 8, "online_head": online,
        "source_head_sha256": head, "files": files, "experiment": experiment}
    if experiment:
        receipt["first_checkpoint_step"] = experiment["first_checkpoint_step"]
    write(directory / "_COMPLETE.json", receipt)
    return directory


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def args(self, phase="warmup"):
        args = train_rl.parser().parse_args([
            "--phase", phase, "--model", str(self.root / "model"), "--head", str(self.root / "head.safetensors"),
            "--head-sha256", "a" * 64, "--train", str(self.root / "train.parquet"),
            "--val", str(self.root / "val.parquet"), "--output-dir", str(self.root / phase),
            "--cache-dir", str(self.root / "cache"), "--upstream-root", str(self.root / "upstream")])
        return args

    def test_phase_historical_argv_and_environment(self):
        for name, stop, save, first in (("warmup", 50, 50, 50), ("online", 100, 50, 51), ("low-lr", 200, 25, 101)):
            args = self.args(name)
            args.bootstrap_checkpoint = self.root / "global_step_50"
            args.resume = None if name == "warmup" else self.root / f"global_step_{train_rl.PHASES[name]['start']}"
            lr = {"start_step": 100, "parent_actor_lr": 1e-6, "actor_lr": 2e-7, "parent_marker_sha256": "b" * 64}
            experiment = {"actor_lr_phase": lr}
            values = train_rl.build_overrides(args)
            self.assertEqual(values["trainer.total_training_steps"], stop)
            self.assertEqual(values["trainer.save_freq"], save)
            for key, value in {"data.train_batch_size": 64, "actor_rollout_ref.rollout.n": 8,
                "trainer.n_gpus_per_node": 8, "trainer.nnodes": 1,
                "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
                "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
                "actor_rollout_ref.rollout.mode": "sync", "actor_rollout_ref.actor.optim.lr_warmup_steps": 5,
                "actor_rollout_ref.model.use_remove_padding": False}.items():
                self.assertEqual(values[key], value)
            inherited = {"RANK": "7", "WORLD_SIZE": "8", "TORCHELASTIC_RUN_ID": "old", "RAY_ADDRESS": "remote",
                "TVG_RL_ACTOR_LR_PHASE": "stale", "PYTHONPATH": "/unpatched", "HOME": "home"}
            env = train_rl.build_environment(args, experiment, self.root / "copy", inherited)
            self.assertFalse(any(k in env for k in ("RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID", "RAY_ADDRESS")))
            self.assertEqual(env["TVG_RL_FIRST_CHECKPOINT_STEP"], str(first))
            self.assertEqual(env["HOME"], "home")
            self.assertNotIn("/unpatched", env["PYTHONPATH"])
            self.assertIn(str(self.root / "copy/rl/verl"), env["PYTHONPATH"])
            self.assertIn(str(self.root / "copy/rl"), env["PYTHONPATH"])
            self.assertEqual("TVG_RL_ACTOR_LR_PHASE" in env, name == "low-lr")
            self.assertEqual("TVG_RL_REWARD_VARIANT" in env, name == "low-lr")
            self.assertEqual(env["TVG_RL_ACTOR_PERFORMANCE"], "1" if name == "low-lr" else "0")
            if name == "low-lr":
                self.assertEqual(env["TVG_RL_REWARD_VARIANT"], "recall_batch_global_f1_tiou")
                self.assertEqual(json.loads(env["TVG_RL_ACTOR_LR_PHASE"]), lr)
            command = train_rl.command_for(values)
            self.assertEqual(command[1:4], ["-u", "-m", "verl.trainer.main_ppo"])
            self.assertFalse(args.execute)
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(runtime.first_online_checkpoint(first), name != "warmup")
                self.assertFalse(runtime.first_online_checkpoint(first + 1))
                self.assertEqual(runtime.worker_environment()["PYTHONPATH"], env["PYTHONPATH"])

    def test_safe_hydra_paths_and_eight_worker_guard(self):
        value = '/local/model with spaces,equals=and"quotes'
        self.assertEqual(json.loads(train_rl.hydra_value(value)), value)
        with self.assertRaises(ValueError):
            train_rl.hydra_value("${oc.env:SECRET}")
        args = self.args()
        args.workers = 4
        with self.assertRaisesRegex(ValueError, "eight"):
            train_rl.plan(args)

    def test_complete_marker_and_identity_guards(self):
        experiment = {"first_checkpoint_step": 51, "phase": "online"}
        directory = marker(self.root, 100, online=True, experiment=experiment)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.verify_complete(directory, True, experiment)["step"], 100)
        with self.assertRaisesRegex(RuntimeError, "different"):
            runtime.verify_complete(directory, True, {"first_checkpoint_step": 101})
        receipt = json.loads((directory / "_COMPLETE.json").read_text())
        receipt["world_size"] = 4
        write(directory / "_COMPLETE.json", receipt)
        with self.assertRaisesRegex(RuntimeError, "world size"):
            runtime.verify_complete(directory)
        receipt["world_size"] = 8
        receipt["files"].pop("actor/optim_world_size_8_rank_7.pt")
        write(directory / "_COMPLETE.json", receipt)
        with self.assertRaisesRegex(RuntimeError, "omits"):
            runtime.verify_complete(directory, True)

    def test_hash_drift_and_traversal(self):
        directory = marker(self.root, 50)
        shard = directory / "actor/model_world_size_8_rank_0.pt"
        shard.write_bytes(b"X" * shard.stat().st_size)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "checksum"):
            runtime.verify_complete(directory)
        with self.assertRaisesRegex(RuntimeError, "receipt path"):
            checkpoint_io.hash_checkpoint_files(directory, ["../outside"])

    def test_visibility_wait_is_bounded(self):
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "size mismatch"):
            checkpoint_io.wait_for_checkpoint_size(self.root / "missing", timeout=0)

    def test_seal_has_explicit_identity_and_first_checkpoint(self):
        experiment = {"phase": "low-lr", "first_checkpoint_step": 101}
        directory = marker(self.root, 101, online=True)
        with patch.dict(os.environ, {"TVG_RL_HEAD_SHA": "a" * 64, "TVG_RL_TRAIN_HEAD": "1",
            "TVG_RL_EXPERIMENT_IDENTITY": json.dumps(experiment), "TVG_RL_FIRST_CHECKPOINT_STEP": "101"}), redirect_stdout(io.StringIO()):
            runtime.checkpoint_receipt(self.root, 101)
            receipt = runtime.verify_complete(directory, True, experiment)
        self.assertEqual(receipt["experiment"], experiment)
        self.assertEqual(receipt["first_checkpoint_step"], 101)
        self.assertEqual(len([n for n in receipt["files"] if n.startswith("actor/model_")]), 8)

    def test_output_never_overwrites_unrelated_or_later_checkpoints(self):
        output = self.root / "output"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "not owned"):
            train_rl.validate_output(output, {}, None)
        experiment = {"phase": "low-lr"}
        write(output / "workflow-plan.json", {"experiment": experiment})
        with self.assertRaisesRegex(ValueError, "explicit"):
            train_rl.validate_output(output, experiment, None)
        checkpoint = marker(output, 125)
        train_rl.validate_output(output, experiment, checkpoint)
        (output / "global_step_150").mkdir()
        with self.assertRaisesRegex(ValueError, "later or incomplete"):
            train_rl.validate_output(output, experiment, checkpoint)

    def row_file(self, name, uid, video_bytes):
        video = self.root / (name + ".mp4")
        video.write_bytes(video_bytes)
        row = {"request_id": uid, "video_path": video.name, "query": "open the door",
            "source_duration_s": 10, "gt_intervals": [[1, 3]]}
        path = self.root / (name + ".jsonl")
        write(path, row)
        return path, row

    def test_schema_prompt_optional_index_and_disjointness(self):
        from infer import prompt
        a, _ = self.row_file("train", "a", b"train video")
        b, _ = self.row_file("monitor", "b", b"monitor video")
        train, monitor = prepare_rl.prepare_rows(a), prepare_rl.prepare_rows(b)
        self.assertEqual(set(train[0]), {"data_source", "prompt", "videos", "reward_model", "extra_info"})
        self.assertEqual(train[0]["prompt"], [{"role": "user", "content": "<video>" + prompt("open the door")}])
        self.assertEqual(train[0]["videos"][0], {"video": str(self.root / "train.mp4"), **prepare_rl.CONTRACT})
        self.assertIsNone(train[0]["extra_info"]["official_row_index"])
        self.assertEqual(json.loads(train[0]["reward_model"]["ground_truth"]), [[1.0, 3.0]])
        prepare_rl.validate_disjoint(train, monitor)
        duplicate = copy.deepcopy(monitor)
        duplicate[0]["extra_info"]["query_id"] = "a"
        with self.assertRaises(ValueError):
            prepare_rl.validate_disjoint(train, duplicate)
        (self.root / "monitor.mp4").write_bytes(b"train video")
        with self.assertRaisesRegex(ValueError, "identical video bytes"):
            prepare_rl.validate_disjoint(train, monitor)

    def test_exclusions_invalid_gt_and_duplicate_ids(self):
        a, row = self.row_file("train", "a", b"train")
        b, _ = self.row_file("monitor", "b", b"monitor")
        train, monitor = prepare_rl.prepare_rows(a), prepare_rl.prepare_rows(b)
        excluded = self.root / "excluded.jsonl"
        for item in ({"request_id": "a"}, {"video_path": "train.mp4"},
                     {"video_sha256": prepare_rl.sha(self.root / "monitor.mp4")}):
            write(excluded, item)
            with self.assertRaisesRegex(ValueError, "excluded"):
                prepare_rl.validate_disjoint(train, monitor, [excluded])
        for gt in ([], [[3, 1]], [[0, 11]], [[float("nan"), 2]], [[True, 2]]):
            write(a, {**row, "gt_intervals": gt})
            with self.assertRaises(ValueError):
                prepare_rl.prepare_rows(a)
        a.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_rl.prepare_rows(a)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow not installed")
    def test_parquet_roundtrip(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path, _ = self.row_file("train", "a", b"video")
        rows = prepare_rl.prepare_rows(path)
        parquet = self.root / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=prepare_rl.parquet_schema()), parquet)
        self.assertEqual(prepare_rl.load_prepared(parquet), rows)

    def test_plan_all_transitions_own_resume_and_dry_run_no_writes(self):
        args = self.args()
        args.model.mkdir()
        args.upstream_root.mkdir()
        args.head.write_bytes(b"paired bootstrap fixture")
        args.head_sha256 = prepare_rl.sha(args.head)
        args.train.write_bytes(b"train parquet fixture; loader mocked")
        args.val.write_bytes(b"monitor parquet fixture; loader mocked")
        (args.model / "model.safetensors").write_bytes(b"merged fixture")
        write(args.model / "config.json", {})
        write(args.model / "span-head-config.json", {"generator_identity": {"adapter_model_sha256": "d" * 64}})
        source = {"head_sha256": args.head_sha256, "sft_protocol": merge_sft.SOFT_PROTOCOL,
            "sft_target": "verifier_v2_yes_probability", "sft_step": 1888,
            "head_config_sha256": prepare_rl.sha(args.model / "span-head-config.json")}
        write(args.model / "_MERGED.json", {"format_version": 1, "peft_version": "0.17.1",
            "source_identity": source, "files": merge_sft.model_files(args.model)})
        videos = {"train": {"/fixture/train.mp4": "a" * 64}, "monitor": {"/fixture/monitor.mp4": "b" * 64}}
        with patch("train_rl.verify_revision"), patch("train_rl.patched_sources"), \
             patch("train_rl.load_prepared", side_effect=lambda p: [None] * (4608 if p == args.train else 32)), \
             patch("train_rl.validate_disjoint", return_value=videos), redirect_stdout(io.StringIO()):
            warmup, _ = train_rl.plan(args)
            self.assertFalse(args.output_dir.exists())
            self.assertFalse(args.cache_dir.exists())
            lineage = warmup["experiment"]["lineage"]
            self.assertEqual(lineage["video_content_sha256"]["train"], actor_lr_phase.canonical_sha(videos["train"]))
            for name in ("src/tvg_confidence/intervals.py", "src/tvg_confidence/semantic.py", "scripts/video_budget.py"):
                self.assertEqual(lineage["implementation"][name], prepare_rl.sha(ROOT / name))
            cp50 = marker(args.output_dir, 50, args.head_sha256, experiment=warmup["experiment"])
            args.phase, args.output_dir = "online", self.root / "online"
            args.resume = args.bootstrap_checkpoint = cp50
            online, _ = train_rl.plan(args)
            self.assertEqual(online["experiment"]["parent_marker_sha256"], prepare_rl.sha(cp50 / "_COMPLETE.json"))
            cp100 = marker(args.output_dir, 100, args.head_sha256, online=True, experiment=online["experiment"])
            args.phase, args.output_dir, args.resume = "low-lr", self.root / "low-lr", cp100
            low, env = train_rl.plan(args)
            expected_phase = {"start_step": 100, "parent_actor_lr": 1e-6, "actor_lr": 2e-7,
                "parent_marker_sha256": prepare_rl.sha(cp100 / "_COMPLETE.json")}
            self.assertEqual(low["experiment"]["actor_lr_phase"], expected_phase)
            self.assertEqual(low["experiment"]["actor_lr_phase_sha256"], actor_lr_phase.canonical_sha(expected_phase))
            self.assertEqual(json.loads(env["TVG_RL_ACTOR_LR_PHASE"]), expected_phase)
            cp125 = marker(args.output_dir, 125, args.head_sha256, online=True, experiment=low["experiment"])
            write(args.output_dir / "workflow-plan.json", low)
            args.resume = cp125
            own, _ = train_rl.plan(args)
            self.assertEqual(own["experiment"], low["experiment"])
            self.assertFalse(args.cache_dir.exists())
            changed_videos = {**videos, "train": {"/fixture/train.mp4": "c" * 64}}
            with patch("train_rl.validate_disjoint", return_value=changed_videos), self.assertRaisesRegex(ValueError, "lineage"):
                train_rl.plan(args)
            receipt = json.loads((cp125 / "_COMPLETE.json").read_text())
            receipt["experiment"]["lineage"]["train_sha256"] = "0" * 64
            write(cp125 / "_COMPLETE.json", receipt)
            with self.assertRaisesRegex(ValueError, "another experiment"):
                train_rl.plan(args)
            args.resume = cp50
            with self.assertRaisesRegex(ValueError, "phase parent"):
                train_rl.plan(args)

    def test_merged_sft_rejects_head_pairing_and_gtbinary(self):
        base, adapter = self.root / "base", self.root / "sft/generator-adapter"
        base.mkdir()
        adapter.mkdir(parents=True)
        write(base / "config.json", {})
        (base / "model.safetensors").write_bytes(b"base fixture")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter fixture")
        write(adapter / "adapter_config.json", {})
        head = adapter.parent / "span-confidence-head.safetensors"
        config = adapter.parent / "span-head-config.json"
        head.write_bytes(b"head fixture")
        identity = {"adapter_model_sha256": prepare_rl.sha(adapter / "adapter_model.safetensors"),
                    "adapter_config_sha256": prepare_rl.sha(adapter / "adapter_config.json")}
        write(config, {"generator_identity": identity, "protocol_version": merge_sft.SOFT_PROTOCOL,
                       "target": "verifier_v2_yes_probability"})
        source = merge_sft.inspect_sources(base, adapter, head, config)
        self.assertEqual(source["adapter"], identity)
        write(config, {"generator_identity": {}})
        with self.assertRaisesRegex(ValueError, "pairing"):
            merge_sft.inspect_sources(base, adapter, head, config)
        source["sft_protocol"] = "gtbinary"
        write(base / "_MERGED.json", {"format_version": 1, "peft_version": "0.17.1", "source_identity": source})
        with self.assertRaisesRegex(ValueError, "not GTbinary"):
            merge_sft.verify_merged(base, head, prepare_rl.sha(head))

    def test_extra_model_weights_cannot_override_verified_shards(self):
        model = self.root / "merged"
        model.mkdir()
        write(model / "config.json", {})
        shard = model / "model-00001-of-00001.safetensors"
        shard.write_bytes(b"verified shard")
        head = model / "span-confidence-head.safetensors"
        head.write_bytes(b"bootstrap")
        write(model / "span-head-config.json", {})
        source = {"head_sha256": prepare_rl.sha(head), "head_config_sha256": prepare_rl.sha(model / "span-head-config.json"),
                  "sft_protocol": merge_sft.SOFT_PROTOCOL, "sft_target": "verifier_v2_yes_probability"}
        write(model / "_MERGED.json", {"format_version": 1, "peft_version": "0.17.1",
              "source_identity": source, "files": merge_sft.model_files(model)})
        merge_sft.verify_merged(model, head, prepare_rl.sha(head))
        (model / "model.safetensors").write_bytes(b"unrecorded single-file override")
        with self.assertRaisesRegex(ValueError, "inventory"):
            merge_sft.verify_merged(model, head, prepare_rl.sha(head))

    def test_video_content_identity_and_cache_reject_same_size_replacement(self):
        a, _ = self.row_file("train", "a", b"video-one")
        b, _ = self.row_file("monitor", "b", b"monitor")
        train, monitor = prepare_rl.prepare_rows(a), prepare_rl.prepare_rows(b)
        original = prepare_rl.validate_disjoint(train, monitor)
        video = train[0]["videos"][0]
        with patch.dict(os.environ, {"TVG_RL_CACHE": str(self.root / "cache")}):
            cached = Path(runtime.cached_video(video)["video"])
            self.assertEqual(cached.read_bytes(), b"video-one")
            (self.root / "train.mp4").write_bytes(b"video-two")
            changed = prepare_rl.validate_disjoint(train, monitor)
            self.assertNotEqual(original["train"], changed["train"])
            replacement = Path(runtime.cached_video(video)["video"])
            self.assertNotEqual(replacement, cached)
            self.assertEqual(replacement.read_bytes(), b"video-two")
            replacement.write_bytes(b"corrupted")
            self.assertEqual(Path(runtime.cached_video(video)["video"]).read_bytes(), b"video-two")

    def test_actor_lr_override_only_after_full_restore_and_own_resume(self):
        class Optimizer:
            param_groups = [{"lr": 1e-6, "initial_lr": 1e-6}]
        class LambdaLR:
            base_lrs = [1e-6]
            _last_lr = [1e-6]
            last_epoch = 100
            _step_count = 101
            lr_lambdas = [lambda step: 1.0]
            def get_last_lr(self):
                return self._last_lr
            def state_dict(self):
                return {"base_lrs": list(self.base_lrs), "_last_lr": list(self._last_lr),
                    "last_epoch": self.last_epoch, "_step_count": self._step_count}
        parent = marker(self.root, 100, online=True)
        phase = {"start_step": 100, "parent_actor_lr": 1e-6, "actor_lr": 2e-7,
                 "parent_marker_sha256": prepare_rl.sha(parent / "_COMPLETE.json")}
        experiment = {"actor_lr_phase_sha256": actor_lr_phase.canonical_sha(phase), "first_checkpoint_step": 101}
        optimizer, scheduler = Optimizer(), LambdaLR()
        result = actor_lr_phase.restore(optimizer, scheduler, parent / "actor", 0, phase, experiment, self.root)
        self.assertEqual(result["action"], "parent_lr_override")
        self.assertEqual(optimizer.param_groups[0], {"lr": 2e-7, "initial_lr": 2e-7})
        self.assertEqual((scheduler.last_epoch, scheduler._step_count), (100, 101))
        own = marker(self.root, 125, online=True, experiment=experiment)
        result = actor_lr_phase.restore(optimizer, scheduler, own / "actor", 0, phase, experiment, self.root)
        self.assertEqual(result["action"], "own_checkpoint_lr_unchanged")
        optimizer.param_groups[0]["lr"] = 1e-6
        with self.assertRaisesRegex(RuntimeError, "learning rates"):
            actor_lr_phase.restore(optimizer, scheduler, own / "actor", 0, phase, experiment, self.root)
        wrong = {**phase, "actor_lr": 3e-7}
        with self.assertRaisesRegex(RuntimeError, "identity"):
            actor_lr_phase.restore(optimizer, scheduler, own / "actor", 0, wrong, experiment, self.root)

    def test_dynamic_export_manifest_not_frozen_release(self):
        from export_head import CONTRACT, PROTOCOL, SPAN_ROLES
        hf, head = self.root / "hf-model", self.root / "online-head"
        hf.mkdir()
        head.mkdir()
        (hf / "model.safetensors").write_bytes(b"new self-produced actor fixture")
        (head / "span-confidence-head.safetensors").write_bytes(b"new paired head fixture")
        digest = prepare_rl.sha(head / "span-confidence-head.safetensors")
        config = {"protocol_version": PROTOCOL, "training_contract": CONTRACT, "target": CONTRACT["target"],
            "pooling": "mean over exact semantic consumed interval tokens", "semantic_roles": SPAN_ROLES,
            "head_architecture": {"input_dim": 2560, "hidden_dim": 256, "output_dim": 1, "dropout": .1},
            "generator_confidence_stop_gradient": True, "generator_frozen_during_training": False,
            "generator_identity": {"step": 125, "checkpoint_marker_sha256": "c" * 64,
                                   "checkpoint": "/private/run/global_step_125"},
            "head_updates": 75, "bootstrap_step": 50, "head_sha256": digest}
        write(head / "span-head-config.json", config)
        receipt = {"status": "complete", "adapter_loaded": False, "step": 125,
            "checkpoint_marker_sha256": "c" * 64, "model_files": {"model.safetensors": prepare_rl.sha(hf / "model.safetensors")},
            "online_head": {"updates": 75, "head_sha256": digest,
                            "head_config_sha256": prepare_rl.sha(head / "span-head-config.json")}}
        metadata = export_rl.release_manifest(hf, head, receipt)
        self.assertEqual(metadata["step"], 125)
        self.assertEqual(metadata["head"]["updates"], 75)
        self.assertEqual(metadata["model_files"], receipt["model_files"])
        public_config = json.loads((hf / "span-head-config.json").read_text())
        self.assertNotIn("checkpoint", public_config["generator_identity"])
        self.assertEqual(metadata["head"]["config_sha256"], prepare_rl.sha(hf / "span-head-config.json"))
        self.assertEqual(json.loads((head / "span-head-config.json").read_text()), config)
        from infer import model_arguments
        args, actual = model_arguments(hf, None, self.root)
        self.assertEqual(actual, metadata)
        self.assertIn("--full-model-manifest", args)

    def test_export_requires_explicit_trust_before_importing_torch(self):
        from export_head import export_head
        with self.assertRaisesRegex(ValueError, "trust"):
            export_head(self.root, "a" * 64, 100, self.root / "output")


class PatchTests(unittest.TestCase):
    def test_replace_fail_closed(self):
        values = {"a.py": "old\nold\n"}
        with self.assertRaises(ValueError):
            patch_verl.replace(values, "a.py", "old", "new")
        self.assertEqual(values["a.py"], "old\nold\n")
        patch_verl.replace(values, "a.py", "old", "new", count=2)
        self.assertEqual(values["a.py"], "new\nnew\n")

    def test_git_revision_pin_and_cleanliness(self):
        with patch("patch_verl.subprocess.check_output", return_value="wrong\n"):
            with self.assertRaisesRegex(ValueError, "revision"):
                patch_verl.verify_revision(ROOT)
        with patch("patch_verl.subprocess.check_output", side_effect=[patch_verl.REVISION, " M rl/file.py"]):
            with self.assertRaisesRegex(ValueError, "clean"):
                patch_verl.verify_revision(ROOT)

    @unittest.skipUnless(os.getenv("TVG_TEST_OMTG_ROOT"), "set TVG_TEST_OMTG_ROOT to local pinned OMTG checkout")
    def test_real_patches_on_readonly_source_copy_and_idempotence(self):
        upstream = patch_verl.verify_revision(Path(os.environ["TVG_TEST_OMTG_ROOT"]))
        # Only required upstream source files are copied into the read-only test fixture;
        # actual revision/cleanliness was checked above, never spoofed in real launcher.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "read-only"
            for name in patch_verl.PATCH_FILES:
                target = source / "rl/verl" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(upstream / "rl/verl" / name, target)
            before = patch_verl.copy_manifest(source)
            for p in sorted(source.rglob("*"), reverse=True):
                p.chmod(0o500 if p.is_dir() else 0o400)
            source.chmod(0o500)
            try:
                with patch("patch_verl.verify_revision", return_value=source), \
                     patch("patch_verl.source_files", return_value=[Path(name) for name in before]):
                    destination = patch_verl.stage_upstream(source, root / "cache/copy")
                self.assertEqual(patch_verl.copy_manifest(source), before)
                self.assertFalse((source / patch_verl.COPY_MARKER).exists())
                self.assertEqual(patch_verl.patch(destination / "rl/verl"), destination)
                self.assertEqual(patch_verl.patch(destination / "rl/verl"), destination)
                workers = (destination / "rl/verl/verl/workers/fsdp_workers.py").read_text()
                load = workers[workers.index("    def load_checkpoint("):]
                self.assertLess(load.index("self.checkpoint_manager.load_checkpoint("), load.index("restore_span_head(self.actor"))
                self.assertLess(load.index("restore_span_head(self.actor"), load.index("restore_actor_lr_phase(self.actor_optimizer"))
                self.assertIn("if os.environ.get('TVG_RL_ACTOR_LR_PHASE'):", load)
                update = workers[workers.index("metrics = self.actor.update_policy"):]
                self.assertLess(update.index("metrics = self.actor.update_policy"), update.index("update_span_head(self.actor)"))
                actor = (destination / "rl/verl/verl/workers/actor/dp_actor.py").read_text()
                self.assertEqual(actor.count("def _forward_micro_batch_impl("), 1)
                self.assertIn('"logits_to_keep": response_length + 1', actor)
                self.assertIn("trim_micro_batch", actor)
                trainer = (destination / "rl/verl/verl/trainer/ppo/ray_trainer.py").read_text()
                section = trainer[trainer.index("# Operating Mode Selection:"):]
                self.assertLess(section.index("compute_log_prob(batch)"), section.index('with marked_timer("reward"'))
                self.assertIn("training_diagnostics", section)
                main = (destination / "rl/verl/verl/trainer/main_ppo.py").read_text()
                self.assertIn('default_runtime_env.setdefault("env_vars", {}).update(worker_environment())', main)
                ray = (destination / "rl/verl/verl/single_controller/ray/base.py").read_text()
                self.assertIn("env_vars.update(worker_environment())", ray)
                for name in patch_verl.PATCH_FILES:
                    ast.parse((destination / "rl/verl" / name).read_text())
                (destination / "rl/verl/verl/workers/actor/dp_actor.py").write_text(actor + "\n# changed\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    patch_verl.verify_copy(destination)
            finally:
                source.chmod(0o700)
                for p in source.rglob("*"):
                    p.chmod(0o700 if p.is_dir() else 0o600)


if __name__ == "__main__":
    unittest.main()
