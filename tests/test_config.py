"""Layered configuration and the derived catalog."""

from __future__ import annotations

import json
import shutil

import pytest

from benchtwm.algorithms import algorithm_names, get_algorithm
from benchtwm.config import Benchmark, ConfigError, deep_merge, layer, parse_overrides
from benchtwm.paths import CONFIG_ROOT


@pytest.fixture(scope="module")
def bench() -> Benchmark:
    return Benchmark.load()


def test_repository_configs_are_valid(bench):
    assert bench.validate() == []


def test_catalog_is_every_behavior_times_every_tier(bench):
    for env_id, spec in bench.dataset_specs.items():
        slug = bench.environments[env_id]["slug"]
        expected = {f"{slug}/{behavior}/{tier}" for behavior in spec["behaviors"] for tier in spec["tiers"]}
        assert expected == {entry.identifier for entry in bench.datasets if entry.env_id == env_id}


def test_every_registered_algorithm_runs_on_every_dataset(bench):
    for entry in bench.datasets:
        assert {method.algo for method in bench.methods(entry.identifier)} == set(algorithm_names())


def test_layers_apply_in_order_with_tier_sections(bench):
    dev = bench.run_config("icu-sepsis/expert/dev", "raw-model", artifact_root="/art")
    final = bench.run_config("icu-sepsis/expert/final", "raw-model", artifact_root="/art")
    assert dev["offline_train_steps"] == 5000  # protocol
    assert dev["reward_loss_weight"] == 5.0  # algorithm defaults
    assert dev["env_kwargs"] == {} and dev["gamma"] == 1.0  # environment run settings
    assert dev["hidden_dim"] == 256 and dev["cem_horizon"] == 10  # ICU override
    assert (dev["learning_rate"], final["learning_rate"]) == (0.0001, 0.001)  # override tiers
    assert (dev["ope_bootstrap_replicates"], final["ope_bootstrap_replicates"]) == (100, 200)  # protocol tiers
    epicare = bench.run_config("epicare/smart/dev", "raw-model", artifact_root="/art")
    assert epicare["hidden_dim"] == 128 and epicare["env_kwargs"] == {"patient_modifiers": True}


def test_run_identity_paths_and_cli_overrides(bench):
    params = bench.run_config("epicare/smart/dev", "latent_model", artifact_root="/art", seed=7,
                              overrides=parse_overrides(["offline_train_steps=3", "env_kwargs.patient_modifiers=false"]))
    assert (params["algo"], params["method_id"], params["seed"]) == ("latent_model", "latent-model", 7)
    assert params["dataset_path"] == "/art/datasets/epicare/smart/dev/train.npz"
    assert params["ope_dataset_path"] == "/art/datasets/epicare/smart/dev/ope_test.npz"
    assert params["offline_train_steps"] == 3 and params["env_kwargs"] == {"patient_modifiers": False}


def test_every_run_config_builds_its_algorithm(bench):
    for entry in bench.datasets:
        for method in bench.methods(entry.identifier):
            params = bench.run_config(entry.identifier, method.method_id, artifact_root="/art")
            assert get_algorithm(params["algo"]).name == method.algo
            assert params["early_stopping_patience"] == 20


def test_dataset_config_carries_behavior_options(bench):
    config = bench.dataset_config("icu-sepsis/epsilon-expert/final", artifact_root="/art")
    assert config["behavior_policy"] == "epsilon_expert" and config["epsilon"] == 0.2
    assert (config["num_train_episodes"], config["num_ope_episodes"]) == (100000, 20000)
    assert config["output_dir"] == "/art/datasets/icu-sepsis/epsilon-expert/final"


def test_unknown_ids_are_reported(bench):
    with pytest.raises(ConfigError, match="Unknown dataset"):
        bench.dataset("icu-sepsis/nope/dev")
    with pytest.raises(ConfigError, match="Available"):
        bench.method("epicare/smart/dev", "ppo")


def test_validation_catches_broken_configs(tmp_path):
    root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, root)
    (root / "algorithms" / "cql.json").unlink()
    spec = json.loads((root / "datasets" / "epicare.json").read_text())
    spec["behaviors"]["smart"]["behavior_policy"] = "unknown"
    (root / "datasets" / "epicare.json").write_text(json.dumps(spec))
    (root / "overrides" / "epicare").mkdir()
    (root / "overrides" / "epicare" / "ppo.json").write_text("{}")
    errors = Benchmark.load(root).validate()
    assert any("configs/algorithms/cql.json" in error for error in errors)
    assert any("undeclared behavior policy" in error for error in errors)
    assert any("unknown algorithm 'ppo'" in error for error in errors)


def test_layer_and_merge_helpers():
    assert layer({"a": 1, "description": "x", "tiers": {"dev": {"a": 2, "b": {"c": 3}}}}, "dev") == {"a": 2, "b": {"c": 3}}
    assert deep_merge({"env_kwargs": {"x": 1, "y": 2}}, {"env_kwargs": {"y": 3}}) == {"env_kwargs": {"x": 1, "y": 3}}
    with pytest.raises(ConfigError):
        parse_overrides(["no_equals_sign"])
