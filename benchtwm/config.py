"""Benchmark configuration: environments, datasets, and layered run configs.

``configs/`` holds four kinds of files, each small and flat:

``protocol.json``                   offline protocol shared by every run
``algorithms/<algo>.json``          default hyperparameters of one algorithm
``envs/<env_id>.json``              environment identity and run settings
``datasets/<env_id>.json``          behavior policies x size tiers of one environment
``overrides/<env_id>/<algo>.json``  optional per-environment deviations

A run config is resolved as protocol -> algorithm -> environment run
settings -> override -> CLI ``--set``. Within any layer, a ``tiers`` object
holds values that apply only to the dataset's tier. Datasets and compatible
methods are derived from these files and the algorithm registry; there is no
hand-maintained list of (dataset, method) pairs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchtwm.algorithms import algorithms, get_algorithm
from benchtwm.envs import environment_ids, get_environment
from benchtwm.paths import CONFIG_ROOT

LAYER_METADATA = {"description", "tiers"}
TIERS = ("dev", "final")


class ConfigError(ValueError):
    """Raised for invalid configuration files or unknown dataset/method ids."""


@dataclass(frozen=True)
class DatasetEntry:
    identifier: str
    env_id: str
    behavior: str
    tier: str
    behavior_policy: str
    description: str
    train_episodes: int
    ope_enabled: bool


@dataclass(frozen=True)
class MethodEntry:
    method_id: str
    algo: str
    family: str
    description: str


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def layer(payload: dict[str, Any], tier: str) -> dict[str, Any]:
    """Parameters of one config layer for ``tier``."""

    params = {key: value for key, value in payload.items() if key not in LAYER_METADATA}
    return deep_merge(params, payload.get("tiers", {}).get(tier, {}))


def parse_overrides(raw: list[str] | None) -> dict[str, Any]:
    """Parse ``key=json-value`` CLI overrides; dotted keys address nested objects."""

    parsed: dict[str, Any] = {}
    for item in raw or []:
        if "=" not in item:
            raise ConfigError(f"Override must use key=json-value, got {item!r}")
        key, raw_value = item.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Override {key!r} must contain a JSON value: {exc.msg}") from exc
        parts = key.split(".")
        if not all(parts):
            raise ConfigError(f"Invalid override key {key!r}")
        target = parsed
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return parsed


def dataset_paths(dataset_id: str, artifact_root: str | Path) -> dict[str, Path]:
    root = Path(artifact_root) / "datasets" / dataset_id
    return {"root": root, "train": root / "train.npz", "validation": root / "val.npz", "ope": root / "ope_test.npz", "metadata": root / "metadata.json"}


class Benchmark:
    def __init__(self, root: str | Path = CONFIG_ROOT):
        self.root = Path(root)
        self.protocol = _read(self.root / "protocol.json")
        self.environments = {path.stem: _read(path) for path in sorted((self.root / "envs").glob("*.json"))}
        self.dataset_specs = {path.stem: _read(path) for path in sorted((self.root / "datasets").glob("*.json"))}
        self._datasets: dict[str, DatasetEntry] = {}
        for env_id, spec in self.dataset_specs.items():
            slug = self.environments.get(env_id, {}).get("slug", env_id)
            for behavior, options in spec.get("behaviors", {}).items():
                for tier, counts in spec.get("tiers", {}).items():
                    identifier = f"{slug}/{behavior}/{tier}"
                    self._datasets[identifier] = DatasetEntry(
                        identifier=identifier,
                        env_id=env_id,
                        behavior=behavior,
                        tier=tier,
                        behavior_policy=str(options["behavior_policy"]),
                        description=str(options.get("description", "")),
                        train_episodes=int(counts["num_train_episodes"]),
                        ope_enabled=bool(spec.get("ope_enabled", False)),
                    )

    @classmethod
    def load(cls, root: str | Path | None = None) -> "Benchmark":
        return cls(root or CONFIG_ROOT)

    # -- discovery ------------------------------------------------------
    @property
    def datasets(self) -> list[DatasetEntry]:
        return sorted(self._datasets.values(), key=lambda item: item.identifier)

    def dataset(self, identifier: str) -> DatasetEntry:
        try:
            return self._datasets[identifier]
        except KeyError as exc:
            raise ConfigError(f"Unknown dataset {identifier!r}. Available: {', '.join(sorted(self._datasets))}") from exc

    def methods(self, dataset_id: str | None = None) -> list[MethodEntry]:
        action_type = None if dataset_id is None else self.environments[self.dataset(dataset_id).env_id]["action_type"]
        return [
            MethodEntry(method_id=cls.method_id, algo=cls.name, family=cls.family, description=cls.description)
            for cls in algorithms()
            if action_type is None or action_type in cls.action_types
        ]

    def method(self, dataset_id: str, method_id: str) -> MethodEntry:
        compatible = self.methods(dataset_id)
        for entry in compatible:
            if method_id in {entry.method_id, entry.algo}:
                return entry
        choices = ", ".join(entry.method_id for entry in compatible)
        raise ConfigError(f"Method {method_id!r} is not available for {dataset_id!r}. Available: {choices}")

    # -- resolution -----------------------------------------------------
    def _environment_run(self, env_id: str) -> dict[str, Any]:
        return {"env_id": env_id, **self.environments[env_id].get("run", {})}

    def dataset_config(self, dataset_id: str, *, artifact_root: str | Path, force: bool = False) -> dict[str, Any]:
        entry = self.dataset(dataset_id)
        spec = self.dataset_specs[entry.env_id]
        options = {key: value for key, value in spec["behaviors"][entry.behavior].items() if key != "description"}
        return {
            **self._environment_run(entry.env_id),
            "dataset_id": entry.identifier,
            "seed": int(spec["seed"]),
            "ope_enabled": entry.ope_enabled,
            **spec["tiers"][entry.tier],
            **options,
            "output_dir": str(dataset_paths(entry.identifier, artifact_root)["root"]),
            "overwrite": bool(force),
        }

    def run_config(
        self,
        dataset_id: str,
        method_id: str,
        *,
        artifact_root: str | Path,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset = self.dataset(dataset_id)
        method = self.method(dataset_id, method_id)
        tier = dataset.tier
        config = layer(self.protocol, tier)
        config = deep_merge(config, layer(self._algorithm_file(method.algo), tier))
        config = deep_merge(config, self._environment_run(dataset.env_id))
        override = self.root / "overrides" / dataset.env_id / f"{method.algo}.json"
        if override.exists():
            config = deep_merge(config, layer(_read(override), tier))
        paths = dataset_paths(dataset.identifier, artifact_root)
        config.update(
            {
                "algo": method.algo,
                "method_id": method.method_id,
                "dataset_id": dataset.identifier,
                "seed": int(0 if seed is None else seed),
                "dataset_path": str(paths["train"]),
                "val_dataset_path": str(paths["validation"]),
                "ope_enabled": dataset.ope_enabled,
            }
        )
        if dataset.ope_enabled:
            config["ope_dataset_path"] = str(paths["ope"])
        return deep_merge(config, overrides or {})

    def _algorithm_file(self, algo: str) -> dict[str, Any]:
        path = self.root / "algorithms" / f"{algo}.json"
        if not path.exists():
            raise ConfigError(f"Missing default hyperparameters for algorithm {algo!r}: {path}")
        return _read(path)

    # -- validation -----------------------------------------------------
    def validate(self) -> list[str]:
        errors: list[str] = []
        registered = set(environment_ids())
        for env_id in sorted(set(self.environments) | set(self.dataset_specs)):
            if env_id not in registered:
                errors.append(f"configs reference unregistered environment {env_id!r}")
            if env_id not in self.environments:
                errors.append(f"missing configs/envs/{env_id}.json")
            if env_id not in self.dataset_specs:
                errors.append(f"missing configs/datasets/{env_id}.json")
        for env_id, spec in self.dataset_specs.items():
            if env_id not in registered:
                continue
            declared = set(get_environment(env_id).behavior_policy_ids)
            for behavior, options in spec.get("behaviors", {}).items():
                if options.get("behavior_policy") not in declared:
                    errors.append(f"dataset {env_id}/{behavior} uses undeclared behavior policy {options.get('behavior_policy')!r}")
            for tier in spec.get("tiers", {}):
                if tier not in TIERS:
                    errors.append(f"configs/datasets/{env_id}.json has unknown tier {tier!r}")
        for cls in algorithms():
            if not (self.root / "algorithms" / f"{cls.name}.json").exists():
                errors.append(f"missing configs/algorithms/{cls.name}.json")
        for path in sorted((self.root / "overrides").glob("*/*.json")):
            if path.parent.name not in self.environments:
                errors.append(f"override {path} targets unknown environment {path.parent.name!r}")
            try:
                get_algorithm(path.stem)
            except KeyError:
                errors.append(f"override {path} targets unknown algorithm {path.stem!r}")
        return errors
