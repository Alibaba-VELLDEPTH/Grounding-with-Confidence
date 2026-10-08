"""CPU-only contract tests; all videos, weights and scores here are test fixtures."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tokenizers import Tokenizer, models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_sft as preparation
import train_sft as trainer
from tvg_confidence.common import read_jsonl, sha256_file, write_json, write_jsonl


class SFTPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="sft-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        base = cls.root / "base"
        base.mkdir()
        alphabet = list(dict.fromkeys('0123456789.,[]-<>|im_end\n '))
        tokenizer = Tokenizer(models.BPE(
            vocab={token: index for index, token in enumerate(["[UNK]", *alphabet])},
            merges=[], unk_token="[UNK]",
        ))
        tokenizer.add_special_tokens(["<|im_end|>"])
        tokenizer.save(str(base / "tokenizer.json"))
        cls.tokenizer = tokenizer
        write_json(base / "config.json", {"model_type": "qwen3_vl", "text_config": {"hidden_size": 2560}})
        write_json(base / "tokenizer_config.json", {"chat_template": "test fixture only"})
        write_json(base / "preprocessor_config.json", {"test_fixture": True})
        (base / "model.safetensors").write_bytes(b"test fixture, not runnable model weights")
        video = cls.root / "video.mp4"
        video.write_bytes(b"test fixture, not a decodable video")
        video_sha = sha256_file(video)
        cls.units, cls.replays, cls.teachers = [], [], []
        candidates = [[2.0, 4.0], [1.8, 3.8], [6.0, 7.0], [8.0, 9.0], [4.5, 5.5]]
        for index in range(1000):
            query = f"fixture event {index}"
            request_id = f"source-{index}"
            unit = {
                "coverage_unit_id": f"unit-{index}", "gt": [[2.0, 4.0]], "original_gt": [[2.0, 4.0]],
                "proposals": [[1.8, 3.8]], "source_ids": [f"zz-{index}", request_id],
                "meta": {"query": query, "source_duration_s": 10.0,
                         "source_video_uid": "fixture-video", "video_path": str(video), "video_sha256": video_sha},
            }
            replay = {
                "request_id": request_id, "source_video_uid": "fixture-video", "video_sha256": video_sha,
                "query": query, "gt_intervals": [[2.0, 4.0]], "prompt_text": f"Find {query}",
                "direct_target_override": json.dumps(candidates),
                "direct_candidate_slots": [{"candidate_id": f"candidate-{index}-{j}", "slot_role": "offline_model_proposal"}
                                           for j in range(5)],
            }
            cls.units.append(unit)
            cls.replays.append(replay)
            for j, score in enumerate([0.12, 0.31, 0.72, 0.88, 0.46]):
                cls.teachers.append({
                    "candidate_id": f"candidate-{index}-{j}", "request_id": request_id,
                    "source_video_uid": "fixture-video", "video_sha256": video_sha, "query": query,
                    "candidate_interval_source_s": candidates[j], "teacher_yes_probability": score,
                })
        write_jsonl(cls.root / "units.jsonl", cls.units)
        write_jsonl(cls.root / "replay.jsonl", cls.replays)
        write_jsonl(cls.root / "teachers.jsonl", cls.teachers)
        cls.args = argparse.Namespace(
            units=cls.root / "units.jsonl", replay=cls.root / "replay.jsonl",
            teacher_scores=cls.root / "teachers.jsonl", base_model=base,
            model_revision=preparation.BASE_REVISION, tokenizer=base / "tokenizer.json",
            expected_tokenizer_sha256=sha256_file(base / "tokenizer.json"),
            output_dir=cls.root / "prepared", world_size=8, gt_rounding_tolerance_s=0.0,
        )
        command = [sys.executable, str(ROOT / "scripts" / "prepare_sft.py")]
        for key, value in vars(cls.args).items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        result = subprocess.run(command, check=True, text=True, capture_output=True,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        cls.report = json.loads(result.stdout)
        cls.manifest_path = cls.args.output_dir / "input-manifest.json"
        cls.manifest = trainer.load_manifest(cls.manifest_path, cls.report["manifest_sha256"])
        cls.rows = list(read_jsonl(cls.args.output_dir / "input.jsonl"))

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.temporary_case.cleanup)
        self.case = Path(self.temporary_case.name)

    def changed_args(self, **changes):
        args = deepcopy(self.args)
        args.output_dir = self.case / "prepared"
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def test_materialized_manifest_and_hashes(self):
        manifest = self.manifest
        self.assertEqual(manifest["protocol_version"], trainer.BOUNDED_POOL_SOFT_CONFIDENCE_V13_PROTOCOL)
        self.assertEqual(manifest["input"]["rows"], 1000)
        self.assertEqual(manifest["input"]["sha256"], sha256_file(Path(manifest["input"]["path"])))
        confidence = manifest["confidence_supervision"]
        self.assertEqual(confidence["teacher_sha256"], sha256_file(Path(confidence["teacher_path"])))
        self.assertEqual(confidence["kind"], "inline_verifier_pyes")
        self.assertFalse(confidence["hard_confidence_labels"])
        self.assertIsNone(manifest["generator_initialization"])
        self.assertTrue(manifest["base_initialization_contract"]["fresh_lora"])
        self.assertEqual(manifest["lora"]["rank"], 32)
        self.assertEqual(manifest["lora"]["alpha"], 64)
        contract = manifest["training_contract"]
        self.assertEqual(contract["target_global_batch_queries"], 64)
        self.assertEqual(contract["epochs_per_arm"], 2)
        self.assertEqual(contract["target_optimizer_steps"], 32)
        self.assertEqual(contract["generator_lr"], 2e-5)
        self.assertEqual(contract["head_lr"], 2e-4)
        self.assertEqual(contract["lambda_conf"], 0.25)
        self.assertEqual(contract["confidence_ranking_weight"], 0.25)
        self.assertFalse(contract["confidence_backprop_to_generator"])
        self.assertEqual(manifest["head_initialization"]["architecture"]["output_dim"], 1)
        self.assertEqual(list(read_jsonl(Path(confidence["teacher_path"]))), self.teachers)
        for name, digest in manifest["model"]["files_sha256"].items():
            self.assertEqual(sha256_file(self.args.base_model / name), digest)
        self.assertEqual((self.args.output_dir / "input-manifest.sha256").read_text().split()[0],
                         self.report["manifest_sha256"])
        self.assertTrue(all(self.report["data"]["checks"].values()))
        self.assertFalse(manifest["paper_main_eligible"])

    def test_prepare_records_compatibility_and_unchanged_scores(self):
        records, summary = trainer.prepare_records(self.manifest)
        self.assertEqual(len(records), 1000)
        self.assertEqual(summary["detached_confidence_replay_queries"], 1000)
        self.assertEqual(summary["endpoint_count"], 5000)
        self.assertEqual(summary["positive_target05"], 2000)
        self.assertEqual(summary["negative_target05"], 3000)
        row, record = self.rows[0], records[0]
        self.assertEqual(row["confidence_replay_source_request_id"], "source-0")
        self.assertEqual(row["direct_lm_masked_char_spans"], [])
        self.assertTrue(row["direct_lm_weighted_char_spans"])
        self.assertEqual(json.loads(record["direct_target"])[0], [2.0, 4.0])
        self.assertEqual(record["confidence_replay_target"], self.replays[0]["direct_target_override"])
        self.assertEqual([target["max_direct_tiou"] for target in record["rollout_targets"]],
                         [0.12, 0.31, 0.72, 0.88, 0.46])
        for target, interval in zip(record["rollout_targets"], json.loads(record["confidence_replay_target"])):
            left, right = target["span_char_span"]
            self.assertEqual(json.loads(record["confidence_replay_target"][left:right]), interval)
        for target_key, count_key, hash_key, cap in (
            ("direct_target", "expected_completion_tokens", "expected_completion_sha256", 480),
            ("confidence_replay_target", "expected_replay_tokens", "expected_replay_sha256", 2048),
        ):
            count, digest = preparation.check_completion(self.tokenizer, record[target_key], cap)
            self.assertEqual((record[count_key], record[hash_key]), (count, digest))

    def test_expected_steps_and_query_dataset_padding(self):
        for count, world in ((1, 1), (64, 8), (65, 8), (1000, 8), (60405, 8), (60405, 1)):
            with self.subTest(rows=count, world=world):
                schedule = preparation.training_schedule(count, world)
                records = [{"rollout_confidence_available": True}] * count
                dataset = trainer.QueryDataset(records, "joint_interleaved", world_size=world,
                                               epochs=2, target_global_batch_queries=64)
                self.assertEqual(dataset.target_optimizer_steps, schedule["target_optimizer_steps"])
                self.assertEqual(dataset.gradient_accumulation_steps, schedule["gradient_accumulation_steps"])
                self.assertEqual(dataset.padding_micro_samples, schedule["zero_weight_padding_micro_samples_per_epoch"])
                self.assertTrue(all(row["branch_scale"] == 0 for row in dataset.records if row["padding_micro_sample"]))
        paper = preparation.training_schedule(60405)
        self.assertEqual(paper["target_optimizer_steps"], 1888)
        self.assertEqual(paper["target_optimizer_steps_per_epoch"], 944)
        self.assertEqual(paper["gradient_accumulation_steps"], 16)
        for count, world in ((0, 8), (-1, 8), (65, 3)):
            with self.assertRaises(ValueError):
                preparation.training_schedule(count, world)

    def test_unit_validation(self):
        for field, value in (("gt", []), ("gt", [[2, 4], [2, 4]]), ("gt", [[2, 11]]),
                             ("original_gt", [[2, 4.1]]), ("proposals", [[1, float("nan")]]),
                             ("proposals", [[4, 3]]), ("source_ids", [])):
            with self.subTest(field=field, value=value):
                unit = deepcopy(self.units[0])
                unit[field] = value
                with self.assertRaises(ValueError):
                    preparation.validate_unit(unit, self.root, {}, 0.0)
        for field, value in (("source_duration_s", float("inf")), ("video_sha256", "0" * 64),
                             ("video_path", str(self.case / "absent.mp4")), ("video_contract", {})):
            with self.subTest(field=field):
                unit = deepcopy(self.units[0])
                unit["meta"][field] = value
                with self.assertRaises(ValueError):
                    preparation.validate_unit(unit, self.root, {}, 0.0)
        unit = deepcopy(self.units[0])
        unit["meta"]["video_path"] = "video.mp4"
        unit["gt"] = [[0.0, 4.0]]
        unit["original_gt"] = [[-0.01, 4.0]]
        with self.assertRaisesRegex(ValueError, "rounding tolerance"):
            preparation.validate_unit(unit, self.root, {}, 0.0)
        valid = preparation.validate_unit(unit, self.root, {}, 0.02)
        self.assertEqual(valid["original_gt"], [[-0.01, 4.0]])
        self.assertTrue(Path(valid["meta"]["video_path"]).is_absolute())

    def test_teacher_alignment_and_soft_probability_validation(self):
        teacher = self.teachers[0]
        binding = {key: teacher[key] for key in ("source_video_uid", "video_sha256", "query",
                                                "request_id", "candidate_interval_source_s")}
        for key, value in (("source_video_uid", "wrong-video"), ("query", "wrong query"),
                           ("candidate_interval_source_s", [3.0, 4.0]), ("request_id", "wrong-source"),
                           ("teacher_yes_probability", 0.0), ("teacher_yes_probability", 1.0),
                           ("teacher_yes_probability", float("nan")), ("teacher_yes_probability", None)):
            with self.subTest(key=key, value=value):
                altered = {**teacher, key: value}
                with self.assertRaises(ValueError):
                    preparation.validate_teacher(altered, binding)
        nested = deepcopy(teacher)
        nested["scores"] = {"yes_probability": nested.pop("teacher_yes_probability")}
        self.assertEqual(preparation.validate_teacher(nested, binding), 0.12)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            preparation.validate_teacher({**teacher, "scores": {"yes_probability": 0.7}}, binding)

    def test_missing_and_duplicate_teachers_refused_before_publish(self):
        for teachers, expected in ((self.teachers[1:], "missing teacher"),
                                   (self.teachers + [self.teachers[0]], "duplicate teacher")):
            with self.subTest(expected=expected):
                path = self.case / "teachers.jsonl"
                write_jsonl(path, teachers)
                args = self.changed_args(teacher_scores=path)
                with self.assertRaisesRegex(ValueError, expected):
                    preparation.materialize(args)
                self.assertFalse(args.output_dir.exists())

    def test_misaligned_teacher_refused_before_publish(self):
        teachers = deepcopy(self.teachers)
        teachers[0]["candidate_interval_source_s"] = [2.0, 5.0]
        path = self.case / "teachers.jsonl"
        write_jsonl(path, teachers)
        args = self.changed_args(teacher_scores=path)
        with self.assertRaisesRegex(ValueError, "teacher candidate_interval_source_s mismatch"):
            preparation.materialize(args)
        self.assertFalse(args.output_dir.exists())

    def test_misaligned_replay_refused(self):
        for key, value in (("query", "wrong query"), ("video_sha256", "0" * 64),
                           ("gt_intervals", [[2, 5]]), ("direct_candidate_slots", [])):
            with self.subTest(key=key):
                replay = {**self.replays[0], key: value}
                with self.assertRaises(ValueError):
                    preparation.validate_replay(self.rows[0], replay, self.tokenizer)
        with self.assertRaisesRegex(ValueError, "completion exceeds"):
            preparation.check_completion(self.tokenizer, "[" * 2049, 2048)

    def test_tiny_corpus_does_not_disable_trainer_gates(self):
        path = self.case / "units.jsonl"
        write_jsonl(path, self.units[:1])
        args = self.changed_args(units=path)
        with self.assertRaisesRegex(SystemExit, "joint training data gate failed"):
            preparation.materialize(args)
        self.assertFalse(args.output_dir.exists())

    def test_sha_and_pinned_model_refusal(self):
        with self.assertRaisesRegex(SystemExit, "manifest SHA mismatch"):
            trainer.load_manifest(self.manifest_path, "0" * 64)
        altered = deepcopy(self.manifest)
        altered["input"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(SystemExit, "data SHA mismatch"):
            trainer.prepare_records(altered)
        altered = deepcopy(self.manifest)
        altered["confidence_supervision"]["teacher_sha256"] = "0" * 64
        with self.assertRaisesRegex(SystemExit, "teacher SHA mismatch"):
            trainer.prepare_records(altered)
        with self.assertRaisesRegex(ValueError, "tokenizer SHA mismatch"):
            preparation.materialize(self.changed_args(expected_tokenizer_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "freezes the original"):
            preparation.materialize(self.changed_args(model_revision="main"))

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "existing output"):
            preparation.materialize(self.args)
        self.assertEqual(sha256_file(self.manifest_path), self.report["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
