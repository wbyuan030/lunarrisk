"""Offline orchestration acceptance: no network, GPU execution, or model training."""
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p31_run_baselines", ROOT / "scripts/p31_run_baselines.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def ids_hash(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def metrics_from_predictions(targets, predictions, classes):
    matrix = [[0] * classes for _ in range(classes)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    support = [sum(row) for row in matrix]
    predicted = [sum(row[c] for row in matrix) for c in range(classes)]
    f1 = [2 * matrix[c][c] / (support[c] + predicted[c]) if support[c] + predicted[c] else 0
          for c in range(classes)]
    return {"rows": len(targets), "support": support, "predicted_counts": predicted,
            "confusion": matrix, "accuracy": sum(a == b for a, b in zip(targets, predictions)) / len(targets),
            "macro_f1_full": sum(f1) / classes,
            "macro_f1_supported": statistics.mean(f for f, n in zip(f1, support) if n),
            "nll": 1., "nll_is_infinite": False, "multiclass_brier_sum": 0.5,
            "largest_observed_batch": len(targets)}


class TestSerialBaselines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.outdir = self.root / "outputs"
        self.data, self.stats, self.mapping, self.plan_path = [self.root / name for name in
                                                           ("data.h5", "stats.json", "map.json", "plan.json")]
        self.data.write_bytes(b"offline synthetic data; no HDF5 reads")
        self.stats.write_text("{}")
        self.mapping.write_text("{}")
        self.plan = json.loads((ROOT / "protocols/p31_baseline_eval_plan.json").read_text())
        self.plan["data"].update(bytes=self.data.stat().st_size, sha256=runner.sha256(self.data),
                                 stats_sha256=runner.sha256(self.stats), class_map_sha256=runner.sha256(self.mapping),
                                 train_rows=3, val_rows=2,
                                 train_ordered_ids_sha256=ids_hash(["t0", "t1", "t2"]),
                                 val_ordered_ids_sha256=ids_hash(["v0", "v1"]))
        self.plan_path.write_text(json.dumps(self.plan))
        self.runtime = self.root / "trusted_runtime"
        (self.runtime / "torch").mkdir(parents=True)
        (self.runtime / "torch/__init__.py").write_text("# package-location fixture only\n")
        self.argv = ["--data", str(self.data), "--outdir", str(self.outdir),
                     "--stats", str(self.stats), "--class-map", str(self.mapping),
                     "--plan", str(self.plan_path), "--runtime-directory", str(self.runtime)]
        self.calls = []
        self.mutate = None
        self.subprocess_patch = patch.object(runner.subprocess, "run", side_effect=self.fake_run)
        self.process = self.subprocess_patch.start()
        self.addCleanup(self.subprocess_patch.stop)
        self.checkpoint_patch = patch.object(runner, "inspect_checkpoint", side_effect=lambda path, runtime: runner.read_json(path))
        self.checkpoint_patch.start()
        self.addCleanup(self.checkpoint_patch.stop)

    def invoke(self, apply=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.main(self.argv + (["--apply"] if apply else []))

    def fake_run(self, command, **kwargs):
        # Model execution is replaced completely; generated artifacts follow the
        # real trainer's schema, with deliberately synthetic tiny split metadata.
        args = {command[i]: command[i + 1] for i in range(6, len(command) - 1, 2)}
        task, seed = args["--task"], int(args["--seed"])
        self.calls.append((task, seed))
        directory = Path(args["--outdir"])
        directory.mkdir(exist_ok=False)
        kwargs["stdout"].write(f"synthetic {task} {seed}\n")
        classes = 5 if task == "age" else 49
        config = runner.expected_config(self.plan, task, seed)
        inputs = {"data_bytes": self.plan["data"]["bytes"], "data_sha256": self.plan["data"]["sha256"],
                  "stats_sha256": self.plan["data"]["stats_sha256"],
                  "class_map_sha256": self.plan["data"]["class_map_sha256"]}
        sources = {path: runner.sha256(ROOT / path) for path in runner.SOURCES}
        splits, model_metrics, prior_metrics, prediction_hashes = {}, {}, {}, {}
        for split, targets, ids in (("train", [0, 1, 0], ["t0", "t1", "t2"]),
                                     ("val", [0, 1], ["v0", "v1"])):
            predictions = targets.copy()
            if seed >= 43:
                predictions[0] = 1
            if seed == 44:
                predictions[1] = 0
            model_metrics[split] = metrics_from_predictions(targets, predictions, classes)
            prior_metrics[split] = metrics_from_predictions(targets, [0] * len(targets), classes)
            splits[split] = {"rows": len(targets), "ordered_ids_sha256": ids_hash(ids),
                             "class_counts": [targets.count(c) for c in range(classes)]}
            for kind, chosen in (("", predictions), ("_prior", [0] * len(targets))):
                path = directory / (split + kind + "_predictions.csv")
                with path.open("w", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(["sample_id", "split", "target"] + [f"p_{c}" for c in range(classes)])
                    for sample, target, prediction in zip(ids, targets, chosen):
                        probabilities = ([2/3, 1/3] + [0.] * (classes - 2)) if kind else [
                            0.9 if c == prediction else 0.1 / (classes - 1) for c in range(classes)]
                        writer.writerow([sample, split, target] + probabilities)
                if not kind:
                    prediction_hashes[split] = runner.sha256(path)
        report = {"status": "P31_BASELINE_TRAINVAL", "purpose": "baseline", "mode": "train",
                  "test_evaluated": False, "config": config, "inputs": inputs, "source_sha256": sources,
                  "splits": splits, "metrics": model_metrics, "train_prior_metrics": prior_metrics,
                  "train_prior_probabilities": [2/3, 1/3] + [0.] * (classes - 2),
                  "prediction_sha256": prediction_hashes, "completed_epochs": 6, "best_epoch": 1,
                  "last_weights_sha256": "a" * 64, "best_weights_sha256": "b" * 64,
                  "train_examples_this_process": 18, "peak_process_rss_bytes": 1000,
                  "peak_gpu_allocated_bytes": 100, "peak_gpu_reserved_bytes": 200,
                  "train_window_seconds_this_process": 6., "train_samples_per_second_this_process": 3.}
        checkpoint = {key: copy.deepcopy(report[key]) for key in
                      ("config", "inputs", "source_sha256", "splits", "test_evaluated", "completed_epochs",
                       "best_epoch", "last_weights_sha256", "best_weights_sha256")}
        checkpoint.update(plan_sha256=runner.sha256(self.plan_path),
                          best_metric=model_metrics["val"]["macro_f1_full"], no_improve=5,
                          history=[{"epoch": i, "train_loss": 0.8, "train_rows": 3,
                                    "largest_train_batch": 3, "train_seconds": 1.,
                                    "train_samples_per_second": 3., "val": copy.deepcopy(model_metrics["val"])}
                                   for i in range(1, 7)])
        asset = {**inputs, "full_content_hash_verified_this_invocation": True,
                 "data_revision": self.plan["data"]["revision"]}
        if self.mutate:
            self.mutate(report, checkpoint, directory)
        for name, value in (("report.json", report), ("checkpoint.pt", checkpoint), ("asset_manifest.json", asset)):
            (directory / name).write_text(json.dumps(value))
        return subprocess.CompletedProcess(command, 0)

    def test_serial_order_locked_budgets_and_child_isolation(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.calls, [("age", 42), ("geology", 42), ("age", 43), ("age", 44),
                                      ("geology", 43), ("geology", 44)])
        for call in self.process.call_args_list:
            command = call.args[0]
            self.assertEqual(command[1:3], ["-I", "-c"])
            self.assertEqual(command[4], str(self.runtime))
            for flag, expected in (("--epochs", "50"), ("--batch-size", "16"), ("--workers", "0"),
                                   ("--lr", "0.001"), ("--patience", "5"), ("--precision", "fp32"),
                                   ("--purpose", "baseline"), ("--track", "official_stored"), ("--device", "cuda")):
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["cwd"], ROOT)
            self.assertNotIn("--resume", command)
            self.assertNotIn("--checkpoint", command)
        final = runner.read_json(self.outdir / "aggregate_06.json")
        summary = final["tasks"]["age"]["baselines"]["independent_task_scratch_cnn"]["val"]["accuracy"]
        self.assertEqual(summary, {"per_seed": {"42": 1., "43": 0.5, "44": 0.}, "n": 3,
                                   "mean": 0.5, "sample_std_ddof1": 0.5})
        self.assertEqual(final["tasks"]["age"]["missing_seeds"], [])
        self.assertFalse(final["test_evaluated"])

    def test_default_preview_does_not_create_outputs_or_read_data(self):
        self.data.unlink()
        self.invoke(apply=False)
        self.process.assert_not_called()
        self.assertFalse(self.outdir.exists())

    def test_restart_reuses_verified_runs_without_launching_models(self):
        self.invoke()
        self.calls.clear()
        self.process.reset_mock()
        self.invoke()
        self.process.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_child_failure_preserves_partial_results_and_does_not_proceed(self):
        original = self.fake_run
        def fail_geology(command, **kwargs):
            if command[command.index("--task") + 1] == "geology":
                kwargs["stdout"].write("synthetic child failure\n")
                raise subprocess.CalledProcessError(1, command)
            return original(command, **kwargs)
        self.process.side_effect = fail_geology
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()
        self.assertEqual(self.calls, [("age", 42)])
        partial = runner.read_json(self.outdir / "aggregate_01.json")
        age = partial["tasks"]["age"]["baselines"]["independent_task_scratch_cnn"]["val"]["accuracy"]
        self.assertEqual(age["n"], 1)
        self.assertIsNone(age["sample_std_ddof1"])
        self.assertEqual(partial["tasks"]["geology"]["completed_seeds"], [])
        self.process.reset_mock()
        with self.assertRaisesRegex(ValueError, "incomplete output"):
            self.invoke()
        self.process.assert_not_called()

    def test_first_seed_validation_blocks_every_later_run(self):
        variants = {
            "nonfinite loss": lambda r, c, d: c["history"][0].update(train_loss=float("nan")),
            "wrong split count": lambda r, c, d: r["splits"]["train"].update(rows=4),
            "early interruption": lambda r, c, d: c.update(completed_epochs=1, history=c["history"][:1]),
            "wrong plan": lambda r, c, d: c.update(plan_sha256="0" * 64),
            "wrong configuration": lambda r, c, d: r["config"].update(batch_size=8),
            "wrong source": lambda r, c, d: r.update(source_sha256={"unexpected": "hash"}),
            "wrong checkpoint input": lambda r, c, d: c["inputs"].update(data_sha256="0" * 64),
            "tie selected late": lambda r, c, d: c.update(best_epoch=2),
            "Test evaluated": lambda r, c, d: r.update(test_evaluated=True),
            "wrong checkpoint weights": lambda r, c, d: c.update(best_weights_sha256="c" * 64),
            "nonfinite throughput": lambda r, c, d: r.update(train_samples_per_second_this_process=float("inf")),
            "nonfinite Val NLL": lambda r, c, d: r["metrics"]["val"].update(nll=None, nll_is_infinite=True),
            "wrong prior": lambda r, c, d: r.update(train_prior_probabilities=[0.2] * 5),
            "altered predictions": lambda r, c, d: (d / "val_predictions.csv").write_text("truncated\n"),
        }
        for index, (name, mutation) in enumerate(variants.items()):
            with self.subTest(name=name):
                self.outdir = self.root / f"bad_{index}"
                self.argv[self.argv.index("--outdir") + 1] = str(self.outdir)
                self.calls.clear()
                self.mutate = mutation
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    self.invoke()
                self.assertEqual(self.calls, [("age", 42)])
                self.assertFalse((self.outdir / "age_seed42/completion.json").exists())
                self.assertEqual(runner.read_json(self.outdir / "aggregate_00.json")["status"], "PARTIAL")

    def test_geology_first_seed_must_pass_before_additional_age_seeds(self):
        def mutate(r, c, d):
            if r["config"]["task"] == "geology":
                c["history"][0]["train_loss"] = float("inf")
        self.mutate = mutate
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.invoke()
        self.assertEqual(self.calls, [("age", 42), ("geology", 42)])
        self.assertTrue((self.outdir / "aggregate_01.json").is_file())

    def test_restart_rejects_changed_artifacts_and_never_overwrites(self):
        self.invoke()
        original = (self.outdir / "age_seed42/checkpoint.pt").read_bytes()
        cases = ("checkpoint.pt", "report.json", "val_predictions.csv", "train_prior_predictions.csv")
        for filename in cases:
            path = self.outdir / "age_seed42" / filename
            old = path.read_bytes()
            path.write_bytes(old + b" ")
            self.process.reset_mock()
            with self.subTest(filename=filename), self.assertRaises((ValueError, KeyError)):
                self.invoke()
            self.process.assert_not_called()
            self.assertEqual(path.read_bytes(), old + b" ")
            path.write_bytes(old)
        self.assertEqual((self.outdir / "age_seed42/checkpoint.pt").read_bytes(), original)

    def test_restart_rejects_source_plan_prediction_and_data_hash_mismatch(self):
        self.invoke()
        receipt_path = self.outdir / "age_seed42/completion.json"
        receipt = runner.read_json(receipt_path)
        for field in ("source_sha256", "plan_sha256", "inputs", "config"):
            changed = copy.deepcopy(receipt)
            changed["identity"][field] = "mismatched"
            receipt_path.write_text(json.dumps(changed))
            self.process.reset_mock()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.invoke()
            self.process.assert_not_called()
        receipt_path.write_text(json.dumps(receipt))
        self.data.write_bytes(b"wrong input")
        with self.assertRaisesRegex(ValueError, "data size/SHA256"):
            self.invoke()
        self.process.assert_not_called()

    def test_rejects_unreceipted_run_even_if_report_exists(self):
        directory = self.outdir / "age_seed42"
        directory.mkdir(parents=True)
        (directory / "report.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "incomplete output"):
            self.invoke()
        self.process.assert_not_called()

    def test_aggregate_keeps_full_and_supported_macro_separate_and_missing_null(self):
        metric = {"accuracy": 0.75, "macro_f1_full": 0.04, "macro_f1_supported": 0.8}
        report = {field: {split: metric for split in ("train", "val")}
                  for field in ("metrics", "train_prior_metrics")}
        result = runner.aggregate([("geology", 42, report)])
        geo = result["tasks"]["geology"]
        self.assertEqual(geo["completed_seeds"], [42])
        self.assertEqual(geo["missing_seeds"], [43, 44])
        val = geo["baselines"]["independent_task_scratch_cnn"]["val"]
        self.assertEqual(val["macro_f1_full"]["per_seed"], {"42": 0.04})
        self.assertEqual(val["macro_f1_supported"]["mean"], 0.8)
        age = result["tasks"]["age"]["baselines"]["independent_task_scratch_cnn"]["val"]["accuracy"]
        self.assertEqual(age, {"per_seed": {}, "n": 0, "mean": None, "sample_std_ddof1": None})

    def test_one_new_run_per_invocation_reuses_prefix_and_preserves_budgets(self):
        self.argv += ["--max-new-runs", "1"]
        expected = [("age", 42), ("geology", 42), ("age", 43), ("age", 44),
                    ("geology", 43), ("geology", 44)]
        for count in range(1, 7):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runner.main(self.argv + ["--apply"]), 0)
            progress = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(self.calls, expected[:count])
            self.assertEqual(progress["new_runs"], 1)
            self.assertEqual(progress["completed"], count)
            self.assertEqual(progress["remaining"], [f"{task}_seed{seed}" for task, seed in expected[count:]])
            self.assertEqual(progress["status"], "PARTIAL" if count < 6 else "COMPLETE")
        for call in self.process.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--epochs") + 1], "50")
        self.process.reset_mock()
        self.invoke()
        self.process.assert_not_called()

    def test_source_change_between_runs_stops_sequence(self):
        original_identity = runner.identity_for
        def changing_identity(*args):
            result = original_identity(*args)
            if self.calls:
                result["source_sha256"] = {"concurrent-edit": "new source"}
            return result
        with patch.object(runner, "identity_for", side_effect=changing_identity):
            with self.assertRaisesRegex(ValueError, "source/plan changed"):
                self.invoke()
        self.assertEqual(self.calls, [("age", 42)])
        self.assertFalse((self.outdir / "age_seed42/completion.json").exists())

    def test_plan_budget_change_is_rejected_before_any_execution(self):
        self.plan["training"]["epochs"] = 2
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "locked baseline settings"):
            self.invoke()
        self.process.assert_not_called()
        self.assertFalse(self.outdir.exists())


class TestCheckpointWeights(unittest.TestCase):
    def test_real_checkpoint_reader_rejects_nonfinite_last_or_best_weights(self):
        try:
            import torch
        except ImportError:
            self.skipTest("existing torch runtime needed for checkpoint-reader test")
        runtime = runner.runtime_directory()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            payload = {key: {} for key in ("config", "inputs", "source_sha256", "splits")}
            payload.update(format_version=1, architecture="P31_scratch_CNN_28_64_128_256",
                           plan_sha256="locked", test_evaluated=False, completed_epochs=6,
                           best_epoch=1, best_metric=0.5, no_improve=5, history=[],
                           state_dict={"weight": torch.tensor([1., 2.])},
                           best_state_dict={"weight": torch.tensor([3., 4.])})
            torch.save(payload, path)
            evidence = runner.inspect_checkpoint(path, runtime)
            self.assertNotEqual(evidence["last_weights_sha256"], evidence["best_weights_sha256"])
            for key in ("state_dict", "best_state_dict"):
                changed = copy.deepcopy(payload)
                changed[key]["weight"][0] = float("nan")
                torch.save(changed, path)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "nonfinite checkpoint"):
                    runner.inspect_checkpoint(path, runtime)


if __name__ == "__main__":
    unittest.main()
