"""Versioned benchmark catalog and recipe resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchtwm.paths import PROJECT_ROOT
from benchtwm.utils.json_handling import get_param_iterable
from benchtwm.utils.io import load_json
from benchtwm.utils.methods import benchmark_track, task_id


class CatalogError(ValueError):
    """Raised when a catalog declaration or requested recipe is invalid."""


@dataclass(frozen=True)
class DatasetRecipe:
    identifier: str
    env_id: str
    tier: str
    config_path: Path
    behavior_policy: str
    episodes: int
    ope_enabled: bool
    description: str


@dataclass(frozen=True)
class MethodRecipe:
    identifier: str
    env_id: str
    tier: str
    mode: str
    method_id: str
    algo: str
    config_path: Path
    description: str
    track: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _config_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    return (path if path.is_absolute() else project_root / path).resolve()


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise CatalogError(f"Invalid override key {dotted_key!r}")
    target = mapping
    for part in parts[:-1]:
        existing = target.get(part)
        if existing is None:
            existing = {}
            target[part] = existing
        if not isinstance(existing, dict):
            raise CatalogError(f"Cannot set {dotted_key!r}: {part!r} is not an object")
        target = existing
    target[parts[-1]] = value


def parse_overrides(raw_overrides: list[str] | None) -> dict[str, Any]:
    """Parse ``key=json-value`` CLI overrides without touching source recipes."""

    parsed: dict[str, Any] = {}
    for raw in raw_overrides or []:
        if "=" not in raw:
            raise CatalogError(f"Override must use key=json-value, got {raw!r}")
        key, raw_value = raw.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"Override {key!r} must contain a JSON value: {exc.msg}") from exc
        _set_nested(parsed, key, value)
    return parsed


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(config))

    def merge(target: dict[str, Any], values: dict[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(result, overrides)
    return result


class Catalog:
    """Loads public dataset and method recipes from ``configs/catalog.json``."""

    def __init__(self, payload: dict[str, Any], *, project_root: Path = PROJECT_ROOT):
        self.payload = payload
        self.project_root = project_root
        self.schema_version = int(payload.get("schema_version", 0))
        self._datasets = {
            item["id"]: DatasetRecipe(
                identifier=item["id"],
                env_id=item["env_id"],
                tier=item["tier"],
                config_path=_config_path(item["config"], project_root),
                behavior_policy=item["behavior_policy"],
                episodes=int(item["episodes"]),
                ope_enabled=bool(item.get("ope_enabled", False)),
                description=str(item.get("description", "")),
            )
            for item in payload.get("datasets", [])
        }
        self._recipes = []
        for item in payload.get("recipes", []):
            modes = item.get("modes", [item.get("mode")])
            tiers = item.get("tiers", [item.get("tier")])
            for mode in modes:
                for tier in tiers:
                    if mode is None or tier is None:
                        raise CatalogError(f"Recipe {item.get('id', '<unnamed>')} must declare mode(s) and tier(s)")
                    values = {"mode": str(mode), "tier": str(tier), "method_id": str(item["method_id"])}
                    self._recipes.append(
                        MethodRecipe(
                            identifier=str(item["id"]).format(**values),
                            env_id=item["env_id"],
                            tier=str(tier),
                            mode=str(mode),
                            method_id=item["method_id"],
                            algo=item["algo"],
                            config_path=_config_path(str(item["config"]).format(**values), project_root),
                            description=str(item.get("description", "")),
                            track=str(item.get("track", "")),
                        )
                    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        catalog_path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "catalog.json"
        return cls(_read_json(catalog_path), project_root=PROJECT_ROOT)

    @property
    def datasets(self) -> list[DatasetRecipe]:
        return sorted(self._datasets.values(), key=lambda item: item.identifier)

    @property
    def recipes(self) -> list[MethodRecipe]:
        return sorted(self._recipes, key=lambda item: (item.env_id, item.mode, item.tier, item.method_id))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"catalog schema_version must be 1, got {self.schema_version}")
        if len(self._datasets) != len(self.payload.get("datasets", [])):
            errors.append("dataset ids must be unique")
        recipe_ids = [item.identifier for item in self._recipes]
        if len(set(recipe_ids)) != len(recipe_ids):
            errors.append("recipe ids must be unique")
        for dataset in self.datasets:
            if dataset.env_id not in self.payload.get("environments", {}):
                errors.append(f"dataset {dataset.identifier} references unknown environment {dataset.env_id}")
            if not dataset.config_path.exists():
                errors.append(f"dataset config missing: {dataset.config_path}")
        for recipe in self.recipes:
            if recipe.mode != "offline":
                errors.append(f"recipe {recipe.identifier} has unsupported mode {recipe.mode!r}")
            if recipe.env_id not in self.payload.get("environments", {}):
                errors.append(f"recipe {recipe.identifier} references unknown environment {recipe.env_id}")
            if not recipe.config_path.exists():
                errors.append(f"recipe config missing: {recipe.config_path}")
            if recipe.track and recipe.track not in {"direct_rl", "provisional_planning"}:
                errors.append(f"recipe {recipe.identifier} has unsupported track {recipe.track!r}")
        for dataset in self.datasets:
            if not any(recipe.env_id == dataset.env_id and recipe.tier == dataset.tier for recipe in self.recipes):
                errors.append(f"dataset {dataset.identifier} has no compatible method recipe")
        return errors

    def dataset(self, identifier: str) -> DatasetRecipe:
        try:
            return self._datasets[identifier]
        except KeyError as exc:
            choices = ", ".join(item.identifier for item in self.datasets)
            raise CatalogError(f"Unknown dataset {identifier!r}. Available: {choices}") from exc

    def compatible_methods(self, dataset_id: str, *, mode: str = "offline") -> list[MethodRecipe]:
        dataset = self.dataset(dataset_id)
        return [
            recipe
            for recipe in self.recipes
            if recipe.env_id == dataset.env_id and recipe.tier == dataset.tier and recipe.mode == mode
        ]

    def recipe(self, dataset_id: str, method_id: str, *, mode: str = "offline") -> MethodRecipe:
        compatible = self.compatible_methods(dataset_id, mode=mode)
        matches = [item for item in compatible if item.method_id == method_id]
        if len(matches) == 1:
            return matches[0]
        choices = ", ".join(item.method_id for item in compatible)
        normalized = method_id.replace("_", "-")
        suggestion = next((item.method_id for item in compatible if item.method_id == normalized), None)
        hint = "" if suggestion is None else f" Use public method ID {suggestion!r}."
        raise CatalogError(
            f"Method {method_id!r} is not available for {dataset_id!r} in {mode} mode.{hint} Available: {choices}"
        )

    @staticmethod
    def _materialize(raw: dict[str, Any], *, seed: int | None) -> dict[str, Any]:
        variants = get_param_iterable(raw)
        if not variants:
            raise CatalogError("Recipe expanded to no concrete configurations")
        if seed is None:
            return dict(variants[0])
        matches = [variant for variant in variants if int(variant.get("seed", -1)) == int(seed)]
        if matches:
            return dict(matches[0])
        base = dict(variants[0])
        base["seed"] = int(seed)
        return base

    def dataset_config(self, dataset_id: str, *, artifact_root: str | Path, force: bool = False) -> dict[str, Any]:
        dataset = self.dataset(dataset_id)
        config = self._materialize(load_json(dataset.config_path), seed=None)
        output_dir = Path(artifact_root) / "datasets" / dataset.identifier
        config.update(
            {
                "schema_version": 1,
                "dataset_id": dataset.identifier,
                "dataset_name": dataset.identifier.replace("/", "_"),
                "env_id": dataset.env_id,
                "environment_seed": int(config.get("environment_seed", 1)),
                "env_kwargs": dict(config.get("env_kwargs", {})),
                "output_dir": str(output_dir),
                "overwrite": bool(force),
                "action_type": "discrete",
                "train_mode": "offline",
            }
        )
        config.setdefault("obs_backend", "state_vector")
        config["task_id"] = task_id(config, dataset.identifier)
        return config

    def run_config(
        self,
        dataset_id: str,
        method_id: str,
        *,
        mode: str = "offline",
        artifact_root: str | Path,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], DatasetRecipe, MethodRecipe]:
        dataset = self.dataset(dataset_id)
        recipe = self.recipe(dataset_id, method_id, mode=mode)
        config = self._materialize(load_json(recipe.config_path), seed=seed)
        dataset_dir = Path(artifact_root) / "datasets" / dataset.identifier
        config.update(
            {
                "schema_version": 1,
                "dataset_id": dataset.identifier,
                "recipe_id": recipe.identifier,
                "env_id": dataset.env_id,
                "method_id": recipe.method_id,
                "environment_seed": int(config.get("environment_seed", 1)),
                "env_kwargs": dict(config.get("env_kwargs", {})),
                "train_mode": mode,
                "action_type": "discrete",
                "task_id": task_id({**config, "dataset_id": dataset.identifier, "train_mode": mode}, dataset.identifier),
                "track": str(recipe.track or ""),
            }
        )
        config.setdefault("obs_backend", "state_vector")
        if not config["track"]:
            config["track"] = benchmark_track(config)
        if mode == "offline":
            config.update(
                {
                    "dataset_path": str(dataset_dir / "train.npz"),
                    "val_dataset_path": str(dataset_dir / "val.npz"),
                }
            )
            if dataset.ope_enabled:
                config["ope_dataset_path"] = str(dataset_dir / "ope_test.npz")
                config["ope_enabled"] = bool(config.get("ope_enabled", True))
            else:
                config.pop("ope_dataset_path", None)
                config["ope_enabled"] = False
        resolved = apply_overrides(config, overrides or {})
        # Overrides are part of the resolved experiment identity (not merely
        # an execution detail), so recompute these fields after applying them.
        resolved.setdefault("obs_backend", "state_vector")
        resolved["action_type"] = "discrete"
        resolved["task_id"] = task_id(resolved, dataset.identifier)
        if not resolved.get("track") or resolved.get("track") is False:
            resolved["track"] = benchmark_track(resolved)
        return resolved, dataset, recipe


def dataset_artifact_paths(dataset_id: str, artifact_root: str | Path) -> dict[str, Path]:
    root = Path(artifact_root) / "datasets" / dataset_id
    return {
        "root": root,
        "train": root / "train.npz",
        "validation": root / "val.npz",
        "ope": root / "ope_test.npz",
        "metadata": root / "metadata.json",
    }
