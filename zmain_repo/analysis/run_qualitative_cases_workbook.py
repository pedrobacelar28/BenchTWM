"""Executable config for analysis/qualitative_cases_workbook.ipynb.

Edit CONFIG below and run:

    python analysis/run_qualitative_cases_workbook.py

The script reuses the notebook logic and analysis.qualitative_cases helpers, but
does not require opening Jupyter.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


CONFIG: dict[str, Any] = {
    # Input trajectories. Keys must be the method IDs expected by the notebook.
    "csv_paths": {
        "dqn_offline": "runs/optuna-best-results-100k/dqn_offline_expert_100k/trajectories.csv",
        "ppo_logged_offline": "runs/optuna-best-results-100k/ppo_offline_expert_100k/trajectories.csv",
        "raw_model_discrete_cem_offline": "runs/optuna-best-results-100k/raw_model_discrete_cem_expert_100k/trajectories.csv",
        "reconstructive_model_continuous_cem_offline": "runs/optuna-best-results-100k/reconstructive_model_continuous_cem_expert_100k/trajectories.csv",
    },
    # Kept for metadata; not used when csv_paths is set.
    "manifest_path": "runs/optuna-best-results-100k/manifest.json",
    "dataset_tag": "100k",
    "output_dir": "reports/qual_cases_optuna_best_100k_4models",
    # Use None to use qc.DEFAULT_DYNAMICS_PATH.
    "dynamics_path": None,
    # Case-selection knobs copied from the notebook.
    "horizon": 3,
    "min_support": 1,
    "length_bucket_size": 8,
    "top_k": 4,
    "stratify_by_outcome": False,
    "max_episode_length": None,
    "selection_unit": "eval_seed_initial_state",
    "require_same_initial_state": True,
    "require_same_length_bucket": True,
    "include_identical_model_trajectories": False,
    "preferred_min_median_length": 6,
    "fallback_min_median_length": 4,
    "review_max_episode_length": 10,
    "minimum_divergent_score": 0.40,
    "selection_group_targets": {
        "all_died": 1,
        "all_survived": 1,
        "mixed_outcome": 2,
    },
}

NOTEBOOK_PATH = Path(__file__).with_name("qualitative_cases_workbook.ipynb")

# Cells that contain executable logic in the notebook. Cell 4 is intentionally
# replaced by CONFIG above.
SETUP_CELLS = [0, 3]
CORE_CELLS = [6, 7, 9, 11, 13, 23, 25, 28]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=None, help="Override CONFIG['top_k'].")
    parser.add_argument("--output-dir", type=str, default=None, help="Override CONFIG['output_dir'].")
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH, help="Notebook to reuse.")
    return parser.parse_args()


def _load_notebook(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _exec_cell(nb: dict[str, Any], cell_idx: int, ns: dict[str, Any], notebook_path: Path) -> None:
    cell = nb["cells"][cell_idx]
    if cell.get("cell_type") != "code":
        return
    source = "".join(cell.get("source", []))
    if not source.strip():
        return
    print(f"[execute-cell] {cell_idx}", flush=True)
    exec(compile(source, f"{notebook_path}:cell-{cell_idx}", "exec"), ns)


def _install_config(ns: dict[str, Any], config: dict[str, Any]) -> None:
    resolve_path = ns["resolve_path"]
    qc = ns["qc"]
    csv_paths = {
        method_id: resolve_path(path)
        for method_id, path in config["csv_paths"].items()
    }
    ns.update(
        {
            "manifest_path": resolve_path(config["manifest_path"]),
            "dataset_tag": str(config["dataset_tag"]),
            "methods": list(csv_paths.keys()),
            "horizon": int(config["horizon"]),
            "min_support": int(config["min_support"]),
            "length_bucket_size": int(config["length_bucket_size"]),
            "top_k": int(config["top_k"]),
            "stratify_by_outcome": bool(config["stratify_by_outcome"]),
            "max_episode_length": config["max_episode_length"],
            "selection_unit": str(config["selection_unit"]),
            "require_same_initial_state": bool(config["require_same_initial_state"]),
            "require_same_length_bucket": bool(config["require_same_length_bucket"]),
            "include_identical_model_trajectories": bool(config["include_identical_model_trajectories"]),
            "preferred_min_median_length": int(config["preferred_min_median_length"]),
            "fallback_min_median_length": int(config["fallback_min_median_length"]),
            "review_max_episode_length": int(config["review_max_episode_length"]),
            "minimum_divergent_score": float(config["minimum_divergent_score"]),
            "selection_group_targets": dict(config["selection_group_targets"]),
            "output_dir": resolve_path(config["output_dir"]),
            "dynamics_path": qc.DEFAULT_DYNAMICS_PATH if config["dynamics_path"] is None else resolve_path(config["dynamics_path"]),
            "use_results_root": False,
            "results_root": resolve_path("runs/optuna-best-results-100k"),
            "run_selector": "latest",
            "csv_paths": csv_paths,
        }
    )


def _print_feature_summaries(ns: dict[str, Any]) -> None:
    features_by_method = ns["features_by_method"]
    for method_id, feats in features_by_method.items():
        print(f"[features] {method_id} shape={feats.shape}", flush=True)
        print("  top initial_state:", flush=True)
        print(feats["initial_state"].value_counts().head(5), flush=True)
        print("  top action_seq:", flush=True)
        print(feats["action_seq"].value_counts().head(5), flush=True)


def main() -> int:
    args = _parse_args()
    config = dict(CONFIG)
    if args.top_k is not None:
        config["top_k"] = args.top_k
        # Keep the common case simple: if top_k is overridden, select mixed
        # outcome cases unless the user edits CONFIG explicitly.
        config["selection_group_targets"] = {
            "all_died": 0,
            "all_survived": 0,
            "mixed_outcome": args.top_k,
        }
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    nb = _load_notebook(args.notebook)
    ns: dict[str, Any] = {"__name__": "__main__"}
    try:
        for cell_idx in SETUP_CELLS:
            _exec_cell(nb, cell_idx, ns, args.notebook)
        _install_config(ns, config)
        for cell_idx in CORE_CELLS:
            _exec_cell(nb, cell_idx, ns, args.notebook)
            if cell_idx == 13:
                _print_feature_summaries(ns)
    except Exception:
        traceback.print_exc()
        return 1

    print(f"[done] output_dir={ns['output_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
