"""Regression tests for the clean operational interface."""

from __future__ import annotations

import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.bench import _early_stopping_plan, _offline_preflight, _ope_plan
from src.algorithms.registry import get_policy_loader, get_trainer
from src.operations.artifacts import initial_manifest, run_directory, write_manifest
from src.operations.catalog import Catalog, apply_overrides, parse_overrides
from src.operations.queue import _write_suite_results
from src.utils.io import load_json, save_json


class TestOperations(unittest.TestCase):
    def test_catalog_exposes_both_environments_and_clean_recipes(self):
        catalog = Catalog.load()
        self.assertEqual(catalog.validate(), [])
        self.assertEqual(
            {dataset.env_id for dataset in catalog.datasets},
            {"icu_sepsis", "epicare"},
        )
        self.assertIn("dqn", [recipe.method_id for recipe in catalog.compatible_methods("epicare/smart/dev")])
        self.assertIn("raw-model/continuous-cem", [recipe.method_id for recipe in catalog.compatible_methods("icu-sepsis/expert/dev")])

    def test_catalog_materializes_explicit_paths_and_typed_overrides(self):
        catalog = Catalog.load()
        overrides = parse_overrides(["offline_train_steps=3", "env_kwargs.patient_modifiers=false"])
        params, dataset, recipe = catalog.run_config(
            "epicare/smart/dev",
            "dqn",
            artifact_root="/tmp/bench-artifacts",
            seed=7,
            overrides=overrides,
        )
        self.assertEqual(dataset.identifier, "epicare/smart/dev")
        self.assertEqual(recipe.method_id, "dqn")
        self.assertEqual(params["seed"], 7)
        self.assertEqual(params["offline_train_steps"], 3)
        self.assertFalse(params["env_kwargs"]["patient_modifiers"])
        self.assertIn("/tmp/bench-artifacts/datasets/epicare/smart/dev/train.npz", params["dataset_path"])

    def test_training_plan_exposes_early_stop_monitor_and_ope_target_mode(self):
        params = {
            "algo": "latent_model",
            "train_mode": "offline",
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 0.01,
            "offline_log_interval": 250,
            "ope_enabled": True,
            "ope_dataset_path": "/tmp/ope_test.npz",
            "eval_deterministic": True,
            "ope_bootstrap_replicates": 100,
        }
        early_stop = _early_stopping_plan(params)
        self.assertTrue(early_stop["enabled"])
        self.assertEqual(early_stop["monitor"], "validation_loss")
        self.assertEqual(early_stop["direction"], "minimize")
        self.assertIn("latent", early_stop["meaning"])
        self.assertEqual(_ope_plan(params)["target_policy_mode"], "deployed_deterministic")

    def test_every_public_recipe_resolves_to_a_registered_trainer_and_policy_loader(self):
        catalog = Catalog.load()
        for recipe in catalog.recipes:
            matching_dataset = next(
                dataset
                for dataset in catalog.datasets
                if dataset.env_id == recipe.env_id and dataset.tier == recipe.tier
            )
            with self.subTest(recipe=recipe.identifier):
                params, _, resolved = catalog.run_config(
                    matching_dataset.identifier,
                    recipe.method_id,
                    artifact_root="/tmp/bench-artifacts",
                )
                self.assertEqual(resolved.identifier, recipe.identifier)
                self.assertIsNotNone(get_trainer(params["algo"]))
                self.assertIsNotNone(get_policy_loader(params["algo"]))

    def test_public_offline_recipes_default_to_twenty_validation_checks_of_patience(self):
        catalog = Catalog.load()
        for recipe in catalog.recipes:
            dataset = next(item for item in catalog.datasets if item.env_id == recipe.env_id and item.tier == recipe.tier)
            params, _, _ = catalog.run_config(dataset.identifier, recipe.method_id, artifact_root="/tmp/bench-artifacts")
            self.assertEqual(params["early_stopping_patience"], 20, recipe.identifier)

    def test_clean_run_directory_and_manifest(self):
        params = {"env_id": "epicare", "algo": "dqn", "train_mode": "offline", "seed": 4, "obs_backend": "state_vector"}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_directory(params, dataset_id="epicare/smart/dev", method_id="dqn", root=tmp)
            self.assertIn("runs", run_dir.parts)
            self.assertNotIn("results", str(run_dir))
            manifest = initial_manifest(params, dataset_id="epicare/smart/dev", method_id="dqn", run_dir=run_dir)
            saved = write_manifest(run_dir, manifest, status="planned")
            self.assertEqual(saved["status"], "planned")
            self.assertTrue((run_dir / "run_manifest.json").exists())

    def test_offline_preflight_requires_held_out_ope_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.npz"
            val = root / "val.npz"
            np.savez_compressed(train, marker=np.asarray([1]))
            np.savez_compressed(val, marker=np.asarray([1]))
            checks = _offline_preflight({"dataset_path": str(train), "val_dataset_path": str(val), "ope_dataset_path": str(root / "ope_test.npz"), "ope_enabled": True})
            self.assertFalse(checks["ready"])
            self.assertIn("ope", checks["missing"])

    def test_cli_dry_run_emits_one_json_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                "scripts/bench.py",
                "train",
                "epicare/smart/dev",
                "dqn",
                "--artifact-root",
                tmp,
                "--dry-run",
                "--json",
            ]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["dataset_id"], "epicare/smart/dev")
        self.assertFalse(payload["preflight"]["ready"])

    def test_cli_reports_canonical_hyphenated_method_ids_without_traceback(self):
        command = [
            sys.executable,
            "scripts/bench.py",
            "train",
            "epicare/smart/dev",
            "latent_model",
            "--dry-run",
        ]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Use public method ID 'latent-model'", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_suite_results_collects_run_metrics_and_planning_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue_dir = root / "queues" / "queue-1"
            direct_dir = root / "runs" / "dqn"
            planning_dir = root / "runs" / "raw-model"
            for output_dir, method_id, track, mean_return, mean_success in (
                (direct_dir, "dqn", "direct_rl", 10.0, 0.8),
                (planning_dir, "raw-model", "provisional_planning", 7.0, 0.6),
            ):
                output_dir.mkdir(parents=True)
                save_json(
                    {
                        "run_id": f"run-{method_id}",
                        "dataset_id": "epicare/smart/dev",
                        "task_id": "epicare/smart/dev|obs=state_vector|action=discrete|mode=offline",
                        "method_id": method_id,
                        "track": track,
                        "environment": "epicare",
                        "train_mode": "offline",
                        "seed": 0,
                        "status": "completed",
                    },
                    output_dir / "run_manifest.json",
                )
                save_json(
                    {
                        "dataset_id": "epicare/smart/dev",
                        "task_id": "epicare/smart/dev|obs=state_vector|action=discrete|mode=offline",
                        "method_id": method_id,
                        "track": track,
                        "obs_backend": "state_vector",
                        "action_type": "discrete",
                        "seed": 0,
                    },
                    output_dir / "config.json",
                )
                save_json(
                    {
                        "train_summary": {"total_gradient_steps": 2},
                        "timing_summary": {"optimization_seconds": 1.5, "evaluation_seconds": 0.5},
                        "eval_summary": {"mean_return": mean_return, "mean_success": mean_success, "mean_length": 3.0},
                    },
                    output_dir / "metrics_summary.json",
                )
                save_json({"mean_return": mean_return, "mean_success": mean_success, "mean_length": 3.0}, output_dir / "eval_summary.json")
            queue_dir.mkdir(parents=True)
            manifest = {
                "schema_version": 1,
                "queue_id": "queue-1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:01:00+00:00",
                "failures": 0,
                "max_workers": 1,
                "gpus": [],
                "jobs": [
                    {
                        "name": "dqn-seed-0",
                        "status": "completed",
                        "output_dir": str(direct_dir),
                        "method_id": "dqn",
                        "track": "direct_rl",
                        "dataset_id": "epicare/smart/dev",
                        "task_id": "epicare/smart/dev|obs=state_vector|action=discrete|mode=offline",
                        "seed": 0,
                    },
                    {
                        "name": "raw-model-seed-0",
                        "status": "completed",
                        "output_dir": str(planning_dir),
                        "method_id": "raw-model",
                        "track": "provisional_planning",
                        "dataset_id": "epicare/smart/dev",
                        "task_id": "epicare/smart/dev|obs=state_vector|action=discrete|mode=offline",
                        "seed": 0,
                    },
                ],
            }
            destination = _write_suite_results(queue_dir, manifest)
            payload = load_json(destination)
            self.assertEqual(payload["suite"]["status"], "completed")
            self.assertEqual(payload["counts"]["total"], 2)
            self.assertEqual(payload["best_direct_rl"]["epicare/smart/dev|obs=state_vector|action=discrete|mode=offline"]["method_id"], "dqn")
            self.assertEqual(payload["planning_gain"][0]["best_direct_method_id"], "dqn")
            self.assertAlmostEqual(payload["planning_gain"][0]["planning_gain_return"], -3.0)

    def test_ope_rejects_canonical_write_for_a_counterfactual_target(self):
        command = [
            sys.executable,
            "scripts/bench.py",
            "ope",
            "placeholder",
            "--target-mode",
            "stochastic",
            "--write-canonical",
        ]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--write-canonical is only valid", result.stderr)


if __name__ == "__main__":
    unittest.main()
