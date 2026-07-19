"""Generate summary tables from processed outputs."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.getcwd())

from analysis.utils import find_best, get_reference_stats, save_dataframe
from src.utils.json_handling import get_sorted_dict
from src.utils.methods import method_id, train_mode
from src.envs.factory import environment_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/summary_table.py <json files>")
        raise SystemExit(1)

    rows = []
    for json_file in sys.argv[1:]:
        config = get_sorted_dict(json_file)
        _, params, payload = find_best(config, data="returns", metric="auc", minmax="max")
        if payload is None or params is None:
            continue
        threshold_metric = "steps_to_threshold" if train_mode(params) == "offline" and "steps_to_threshold" in payload["summary"] else "episodes_to_threshold"
        rows.append(
            {
                "env_id": environment_id(params),
                "algo": method_id(params),
                "auc_return": float(payload["summary"]["auc_return"]["mean"]),
                "final_return": float(payload["summary"]["final_return"]["mean"]),
                "best_eval_return": float(payload["summary"].get("best_eval_return", {}).get("mean", payload["summary"]["final_return"]["mean"])),
                "budget_to_threshold": float(payload["summary"][threshold_metric]["mean"]),
                "action_stability_l1": float(payload["summary"]["mean_action_stability_l1"]["mean"]),
                "abrupt_jump_rate": float(payload["summary"]["mean_abrupt_jump_rate"]["mean"]),
            }
        )

    if rows and all(row["env_id"] == "icu_sepsis" for row in rows):
        refs = get_reference_stats()
        rows.append({"env_id": "icu_sepsis", "algo": "expert", "auc_return": refs["expert_policy"]["j_mean"], "final_return": refs["expert_policy"]["j_mean"], "best_eval_return": refs["expert_policy"]["j_mean"], "budget_to_threshold": 0, "action_stability_l1": 0.0, "abrupt_jump_rate": 0.0})
        rows.append({"env_id": "icu_sepsis", "algo": "optimal", "auc_return": refs["pi_star"]["j_mean"], "final_return": refs["pi_star"]["j_mean"], "best_eval_return": refs["pi_star"]["j_mean"], "budget_to_threshold": 0, "action_stability_l1": 0.0, "abrupt_jump_rate": 0.0})
    df = pd.DataFrame(rows)
    save_dataframe(df, "plots/summary_table.csv")
    print(df.to_string(index=False))
