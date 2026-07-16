"""Rerun selected best world-model Optuna trials with trajectory exports enabled.

Edit SELECT_STUDIES below if you only want one model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.run_experiment import run_experiment
from scripts.tune_offline_world_models import AVAILABLE_MODELS, COMMON_OVERRIDES, SEED_BASE, STUDY_DIR


SELECT_STUDIES = [
    "raw_model_discrete_cem_expert_5k",
    "reconstructive_model_continuous_cem_expert_5k",
]

# Use one GPU at a time for deterministic-ish export runs.
GPU = 0
FORCE_RUN = False

EXPORT_OVERRIDES = {
    "save_checkpoints": True,
    "save_trajectories": True,
    "show_progress": True,
    "offline_console_style": "compact",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _storage_url(study_name: str) -> str:
    return f"sqlite:///{STUDY_DIR / f'{study_name}.sqlite3'}"


def _base_params(base_config: Path) -> dict[str, Any]:
    params = _load_json(base_config)
    params.update(COMMON_OVERRIDES)
    params.update(EXPORT_OVERRIDES)
    return params


def _params_from_best_trial(spec: dict[str, Any], trial) -> dict[str, Any]:
    params = _base_params(Path(spec["base_config"]))
    params.update(trial.params)
    params["seed"] = SEED_BASE + int(trial.number)
    if params["cem_elites"] >= params["cem_samples"]:
        params["cem_elites"] = max(1, params["cem_samples"] // 8)
    return params


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

    import optuna

    specs = {str(item["name"]): item for item in AVAILABLE_MODELS.values()}
    for study_name in SELECT_STUDIES:
        if study_name not in specs:
            raise KeyError(f"Unknown study {study_name!r}; available={sorted(specs)}")
        study = optuna.load_study(study_name=study_name, storage=_storage_url(study_name))
        trial = study.best_trial
        params = _params_from_best_trial(specs[study_name], trial)
        print(
            "[export-best] "
            f"study={study_name} trial={trial.number} survival={study.best_value:.6f} "
            f"seed={params['seed']} gpu={GPU}",
            flush=True,
        )
        output_dir = run_experiment(params, force_run=FORCE_RUN)
        print(f"[export-best-done] study={study_name} output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
