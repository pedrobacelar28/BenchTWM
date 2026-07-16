"""Evaluate saved checkpoints under admissible-action perturbations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(os.getcwd())

from analysis.utils import save_dataframe
from src.algorithms.registry import get_policy_loader
from src.bootstrap import bootstrap_icu_helpers
from src.evaluation import evaluate_policy
from src.utils.io import load_json
from src.utils.torch_utils import get_torch_device


def make_perturbed_params(sigma: float, trial_seed: int):
    bootstrap_icu_helpers()
    import numpy as np
    from icu_sepsis.utils.io import MDPParameters
    from icu_sepsis_helpers.policy_examination import perturb

    np.random.seed(trial_seed)
    params_path = Path(__file__).resolve().parents[1] / ".." / "icu-sepsis" / "packages" / "icu_sepsis" / "icu_sepsis" / "envs" / "assets"
    params_path = params_path.resolve()
    params = MDPParameters(params_path)
    tx_mat_new, admissible_actions_new = perturb(params.tx_mat, params.admissible_actions, sigma)
    params._tx_mat = tx_mat_new
    params._admissible_actions = admissible_actions_new
    return params


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/robustness.py <run_dir> [checkpoint]")
        raise SystemExit(1)
    run_dir = Path(sys.argv[1])
    checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "best"
    config = load_json(run_dir / "config.json")
    checkpoint = torch.load(run_dir / f"checkpoint_{checkpoint_name}.pt", map_location=get_torch_device())
    policy = get_policy_loader(config["algo"])(config, checkpoint)

    rows = []
    for sigma in config.get("robustness_sigmas", [0.01, 0.05, 0.2, 0.5, 0.8]):
        for trial in range(int(config.get("robustness_trials", 4))):
            params = make_perturbed_params(float(sigma), int(config["seed"]) + trial)
            result = evaluate_policy(
                policy,
                config=config,
                num_episodes=int(config.get("robustness_eval_episodes", 25)),
                eval_seed_base=int(config["eval_seed_base"]) + 1000 * trial,
                output_dir=None,
                checkpoint_name=checkpoint_name,
                params=params,
                export_csv=False,
            )
            rows.append(
                {
                    "algo": config["algo"],
                    "checkpoint": checkpoint_name,
                    "sigma": float(sigma),
                    "trial": trial,
                    "mean_return": result["summary"]["mean_return"],
                    "mean_survival": result["summary"]["mean_survival"],
                    "mean_length": result["summary"]["mean_length"],
                }
            )
    df = pd.DataFrame(rows)
    save_dataframe(df, run_dir / f"robustness_{checkpoint_name}.csv")
    print(df.groupby("sigma")[["mean_return", "mean_survival", "mean_length"]].mean().to_string())
