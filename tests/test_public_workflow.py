from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate
import infer
import prepare_model_release
from head_checkpoint import CONTRACT, PROTOCOL, SPAN_ROLES
from tvg_confidence.common import sha256_file


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def model_fixture(root):
    model = root / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"synthetic weights, not a runnable model")
    (model / "span-confidence-head.safetensors").write_bytes(b"synthetic head")
    head_sha = sha256_file(model / "span-confidence-head.safetensors")
    config = {
        "protocol_version": PROTOCOL, "training_contract": CONTRACT,
        "target": CONTRACT["target"], "pooling": "mean over exact semantic consumed interval tokens",
        "semantic_roles": SPAN_ROLES,
        "head_architecture": {"input_dim": 2560, "hidden_dim": 256, "output_dim": 1, "dropout": 0.1},
        "generator_confidence_stop_gradient": True, "generator_frozen_during_training": False,
        "head_updates": 150, "bootstrap_step": 50, "head_sha256": head_sha,
        "generator_identity": {"step": 200, "checkpoint_marker_sha256": "a" * 64,
                               "private_path": "/private/example"},
    }
    write_json(model / "span-head-config.json", config)
    write_json(model / "config.json", {"_name_or_path": "/private/example", "hidden_size": 2560})
    release = {"format_version": 1, "base_model_revision": infer.core.EXPECTED_MODEL_REVISION,
               "step": 200, "checkpoint_marker_sha256": "a" * 64,
               "model_files": {name: sha256_file(model / name)
                               for name in ("model.safetensors", "config.json")},
               "head": {"weights": "span-confidence-head.safetensors", "config": "span-head-config.json",
                        "sha256": head_sha, "config_sha256": sha256_file(model / "span-head-config.json"),
                        "updates": 150}}
    write_json(model / "release.json", release)
    return model, release


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.row = {"request_id": "q", "source_duration_s": 10, "gt_intervals": [[1, 3]],
                    "candidate_intervals": [[1, 3], [1.1, 3.1], [7, 8]], "span_scores": [0.1, 0.9, 0.8]}

    def test_nms_precedes_threshold_and_does_not_use_score_order(self):
        report = evaluate.evaluate_rows([self.row], 0.5)
        self.assertEqual(report["per_query"][0]["selected_intervals"], [[7.0, 8.0]])
        self.assertEqual(report["metrics"]["occurrence_recall_05"], 0)

    def test_failed_queries_remain_in_denominator(self):
        rows = evaluate.read_rows(ROOT / "examples/predictions.jsonl")
        report = evaluate.evaluate_rows(rows, 0.5)
        self.assertEqual((report["queries"], report["failed_queries"]), (3, 1))
        self.assertAlmostEqual(report["metrics"]["occurrence_recall_05"], 0.5)
        self.assertAlmostEqual(report["metrics"]["occurrence_f1_05"], 5 / 9)

    def test_request_denominator_and_annotation_identity(self):
        self.assertTrue(evaluate.evaluate_rows([self.row], 0.5, [self.row])["request_identity_verified"])
        other = {**self.row, "request_id": "missing"}
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            evaluate.evaluate_rows([self.row], 0.5, [self.row, other])
        with self.assertRaisesRegex(ValueError, "ground truth mismatch"):
            evaluate.evaluate_rows([self.row], 0.5, [{**self.row, "gt_intervals": []}])

    def test_empty_queries_and_duplicate_indices(self):
        empty = {**self.row, "gt_intervals": [], "candidate_intervals": [], "span_scores": []}
        report = evaluate.evaluate_rows([empty], 1.0)
        self.assertEqual(report["queries"], 1)
        self.assertEqual(report["metrics"]["count_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["occurrence_f1_05"], 1.0)
        self.assertEqual(evaluate.evaluate_rows([empty], 1.0, metric_protocol="set")
                         ["metrics"]["occurrence_f1_05"], 0.0)
        duplicate = {**self.row, "candidate_intervals": [[1, 3], [1, 3], [7, 8]],
                     "span_scores": [0.1, 0.99, 0.8]}
        self.assertEqual(evaluate.select_intervals(duplicate, 0.5), [[7.0, 8.0]])

    def test_official_merge_before_matching_preserves_inputs(self):
        row = {"request_id": "overlap", "source_duration_s": 10,
               "gt_intervals": [[0, 3], [6, 8]],
               "candidate_intervals": [[0, 3], [2, 8]], "span_scores": [0.9, 0.9]}
        original = json.dumps(row)
        official = evaluate.evaluate_rows([row], 0.5)
        legacy = evaluate.evaluate_rows([row], 0.5, metric_protocol="set")
        self.assertEqual(official["metric_protocol"], "omtg")
        self.assertEqual(official["metrics"]["occurrence_f1_05"], 0.0)
        self.assertEqual(legacy["metrics"]["occurrence_f1_05"], 0.5)
        self.assertEqual(official["metrics"]["tiou"], 5 / 8)
        self.assertEqual(official["per_query"][0]["selected_intervals"], [[0, 3], [2, 8]])
        self.assertEqual(json.dumps(row), original)

    def test_invalid_inputs(self):
        for scores in ([0.1], [float("nan"), 0.9, 0.8], [True, 0.9, 0.8], [1.1, 0.9, 0.8]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                evaluate.evaluate_rows([{**self.row, "span_scores": scores}], 0.5)
        for threshold in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                evaluate.evaluate_rows([self.row], threshold)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate.evaluate_rows([self.row, self.row], 0.5)
        with self.assertRaises(ValueError):
            evaluate.evaluate_rows([{**self.row, "gt_intervals": [[3, 1]]}], 0.5)


class InferenceTests(unittest.TestCase):
    def test_preparation_and_required_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "video.mp4").write_bytes(b"synthetic video")
            requests = root / "requests.jsonl"
            row = {"request_id": "q", "video_path": "video.mp4", "query": "opens a door",
                   "source_duration_s": 10}
            write_json(requests, row)
            prepared = infer.prepare_rows(requests)
            self.assertEqual(prepared[0]["video_path"], str((root / "video.mp4").resolve()))
            self.assertEqual(prepared[0]["video_contract"], infer.VIDEO_CONTRACT)
            with self.assertRaisesRegex(ValueError, "requires gt_intervals"):
                infer.prepare_rows(requests, require_gt=True)
            write_json(requests, {**row, "gt_intervals": [[0, 12]]})
            with self.assertRaisesRegex(ValueError, "outside"):
                infer.prepare_rows(requests)

    def test_release_checks_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _ = model_fixture(root)
            arguments, _ = infer.model_arguments(model, None, root)
            self.assertIn("--full-model-manifest", arguments)
            extra = model / "model-00001-of-00001.safetensors"
            extra.write_bytes(b"unrecorded weights")
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                infer.model_arguments(model, None, root)
            extra.unlink()
            head_config = json.loads((model / "span-head-config.json").read_text())
            head_config["generator_identity"]["step"] = 199
            write_json(model / "span-head-config.json", head_config)
            release = json.loads((model / "release.json").read_text())
            release["head"]["config_sha256"] = sha256_file(model / "span-head-config.json")
            write_json(model / "release.json", release)
            with self.assertRaisesRegex(ValueError, "pairing failed"):
                infer.model_arguments(model, None, root)
            (model / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Generator checksum"):
                infer.model_arguments(model, None, root)
            with self.assertRaisesRegex(ValueError, "flat filenames"):
                infer.release_file(model, "../other")

    def test_check_only_does_not_run_inference_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _ = model_fixture(root)
            (root / "video.mp4").write_bytes(b"synthetic video")
            requests = root / "requests.jsonl"
            write_json(requests, {"request_id": "q", "video_path": "video.mp4", "query": "opens a door",
                                  "source_duration_s": 10, "gt_intervals": []})
            output = root / "out"
            argv = ["infer", "--model", str(model), "--input", str(requests),
                    "--output-dir", str(output), "--evaluate", "--check-only"]
            with patch.object(sys, "argv", argv), patch.object(infer.core, "main") as gpu:
                self.assertEqual(infer.main(), 0)
                gpu.assert_not_called()
            self.assertFalse(output.exists())

    def test_sft_adapter_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "generator-adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            write_json(adapter / "adapter_config.json", {})
            (root / "span-confidence-head.safetensors").write_bytes(b"head")
            config = {"generator_identity": {
                "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json")}}
            write_json(root / "span-head-config.json", config)
            arguments, _ = infer.model_arguments(root, root, root)
            self.assertIn("--adapter-dir", arguments)
            config["generator_identity"]["adapter_model_sha256"] = "b" * 64
            write_json(root / "span-head-config.json", config)
            with self.assertRaisesRegex(ValueError, "pairing mismatch"):
                infer.model_arguments(root, root, root)

    def test_infer_then_eval_with_mocked_gpu_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, _ = model_fixture(root)
            (root / "video.mp4").write_bytes(b"synthetic video")
            requests = root / "requests.jsonl"
            write_json(requests, {"request_id": "q", "video_path": "video.mp4", "query": "opens a door",
                                  "source_duration_s": 10, "gt_intervals": [[1, 3]]})
            output = root / "out"

            def fake_core():
                self.assertIn("--candidate-aware-stop", sys.argv)
                self.assertIn("--exact-dedup", sys.argv)
                self.assertIn("--salvage-truncated-interval-prefix", sys.argv)
                rows = evaluate.read_rows(Path(sys.argv[sys.argv.index("--input") + 1]))
                output.mkdir()
                rows[0].update(infer_status="success", candidate_intervals=[[1, 3]],
                               span_predictions=[{"predicted_span_correctness_probability": 0.8}])
                write_json(output / "outputs.jsonl", rows[0])
                return 0

            argv = ["infer", "--model", str(model), "--input", str(requests),
                    "--output-dir", str(output), "--threshold", "0.5", "--evaluate",
                    "--metric-protocol", "set"]
            with patch.object(sys, "argv", argv), patch.object(infer.core, "main", fake_core), \
                    patch.object(infer, "install_into_core"):
                self.assertEqual(infer.main(), 0)
            report = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(report["metric_protocol"], "set")
            self.assertEqual(report["metrics"]["occurrence_f1_05"], 1)
            self.assertTrue(report["request_identity_verified"])
            self.assertEqual(evaluate.read_rows(output / "predictions.jsonl")[0]["selected_intervals"], [[1, 3]])


class ReleaseTests(unittest.TestCase):
    def test_portable_package_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, release = model_fixture(root)
            before = {p.name: p.read_bytes() for p in model.iterdir()}
            (root / "configs").mkdir()
            (root / "docs").mkdir()
            (root / "licenses").mkdir()
            (root / "licenses/TimeLens2.txt").write_text("Synthetic retained license fixture\n")
            write_json(root / "configs/release-model.json", release)
            (root / "docs/model-card.md").write_text("Synthetic model card\n")
            output = root / "release"
            argv = ["prepare", "--model-dir", str(model), "--head-dir", str(model), "--output-dir", str(output)]
            with patch.object(prepare_model_release, "ROOT", root), patch.object(sys, "argv", argv + ["--check-only"]):
                prepare_model_release.main()
            self.assertFalse(output.exists())
            with patch.object(prepare_model_release, "ROOT", root), patch.object(sys, "argv", argv):
                prepare_model_release.main()
            self.assertEqual(before, {p.name: p.read_bytes() for p in model.iterdir()})
            self.assertNotIn("_name_or_path", json.loads((output / "config.json").read_text()))
            self.assertNotIn("private_path", json.loads((output / "span-head-config.json").read_text())["generator_identity"])
            self.assertEqual((output / "model.safetensors").read_bytes(), before["model.safetensors"])
            for line in (output / "MANIFEST.sha256").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, sha256_file(output / name))
            infer.model_arguments(output, None, root)


if __name__ == "__main__":
    unittest.main()
