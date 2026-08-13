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
from src.algorithms.common import TrainLogger
from src.algorithms.registry import get_policy_loader, get_trainer_for_mode
from src.operations.artifacts import initial_manifest, run_directory, write_manifest
from src.operations.catalog import Catalog, apply_overrides, parse_overrides


class TestOperations(unittest.TestCase):
    def test_catalog_exposes_both_environments_and_clean_recipes(self):
        catalog = Catalog.load()
        self.assertEqual(catalog.validate(), [])
        self.assertEqual({dataset.env_id for dataset in catalog.datasets}, {"icu_sepsis", "epicare"})
        self.assertIn("dqn", [recipe.method_id for recipe in catalog.compatible_methods("epicare/smart/dev")])
        self.assertIn("raw-model/discrete-cem", [recipe.method_id for recipe in catalog.compatible_methods("icu-sepsis/expert-5k/dev")])

    def test_catalog_materializes_explicit_paths_and_typed_overrides(self):
        catalog = Catalog.load()
        overrides = parse_overrides(["offline_train_steps=3", "env_kwargs.patient_modifiers=false"])
        params, dataset, recipe = catalog.run_config(
            "epicare/smart/dev",
            "dqn",
            mode="offline",
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
                    mode=recipe.mode,
                    artifact_root="/tmp/bench-artifacts",
                )
                self.assertEqual(resolved.identifier, recipe.identifier)
                self.assertIsNotNone(get_trainer_for_mode(params["algo"], recipe.mode))
                self.assertIsNotNone(get_policy_loader(params["algo"]))

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
        self.assertEqual(payload["mode"], "offline")
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

    def test_online_logger_uses_environment_success_and_action_geometry(self):
        logger = TrainLogger(show_progress=False)
        logger.log_episode(
            1,
            total_reward=3.0,
            episode_length=2,
            total_env_steps=2,
            actions=[0, 15],
            success=False,
            action_distance=lambda previous, current: float(previous != current),
            transition_metrics=[
                {"symptom_burden": 0.8, "remission": 0.0},
                {"symptom_burden": 0.4, "remission": 0.0},
            ],
            outcome={"success": False, "remission": False, "adverse_event": False},
        )
        payload = logger.metrics_payload()
        self.assertEqual(payload["success_rate"].tolist(), [0.0])
        self.assertEqual(payload["survival_rate"].tolist(), [0.0])
        self.assertEqual(payload["action_stability_l1"].tolist(), [1.0])
        self.assertAlmostEqual(payload["environment_metrics"]["mean_symptom_burden"][0], 0.6)
        self.assertEqual(payload["environment_metrics"]["outcome_remission"].tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
