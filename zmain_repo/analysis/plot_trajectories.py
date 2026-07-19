"""Plot one exported trajectory using its environment specification."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.getcwd())

from src.utils.io import ensure_dir


def _spec(run_dir: Path) -> dict:
    path = run_dir / "environment_spec.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"env_id": "icu_sepsis", "metric_names": ["sofa_score"], "action_geometry": {"type": "grid"}}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/plot_trajectories.py <run_dir> [episode_id]")
        raise SystemExit(1)
    run_dir = Path(sys.argv[1])
    episode_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    df = pd.read_csv(run_dir / "trajectories.csv")
    if episode_id is None:
        episode_id = int(df["episode_id"].iloc[0])
    episode = df[df["episode_id"] == episode_id].copy()
    spec = _spec(run_dir)
    metric_columns = [f"metric_{name}" for name in spec.get("metric_names", []) if f"metric_{name}" in episode]
    fig, axes = plt.subplots(2 + len(metric_columns), 1, figsize=(9, 2.5 * (2 + len(metric_columns))), dpi=180, sharex=True)
    axes = list(axes)
    for axis, column in zip(axes, metric_columns):
        axis.plot(episode["t"], episode[column], marker="o")
        axis.set_ylabel(column.removeprefix("metric_"))
    action_axis = axes[-2]
    action_axis.step(episode["t"], episode["action_idx"], where="post")
    action_axis.set_ylabel("action")
    reward_axis = axes[-1]
    reward_axis.bar(episode["t"], episode["reward"])
    reward_axis.set_ylabel("reward")
    reward_axis.set_xlabel("t")
    out_dir = ensure_dir(Path("plots"))
    fig.tight_layout()
    fig.savefig(out_dir / f"trajectory_{spec['env_id']}_episode_{episode_id}.png")
