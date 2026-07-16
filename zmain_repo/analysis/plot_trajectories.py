"""Plot exported ICU-Sepsis trajectories without notebooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.getcwd())

from src.utils.io import ensure_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/plot_trajectories.py <run_dir> [episode_id]")
        raise SystemExit(1)
    run_dir = Path(sys.argv[1])
    episode_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    df = pd.read_csv(run_dir / "trajectories.csv")
    if episode_id is None:
        episode_id = int(df["episode_id"].iloc[0])
    epi = df[df["episode_id"] == episode_id].copy()
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), dpi=200, sharex=True)
    axes[0].plot(epi["t"], epi["sofa_score"], marker="o")
    axes[0].set_ylabel("SOFA")
    axes[1].step(epi["t"], epi["fluid_bin"], where="post")
    axes[1].set_ylabel("Fluid")
    axes[2].step(epi["t"], epi["vaso_bin"], where="post")
    axes[2].set_ylabel("Vaso")
    axes[2].set_xlabel("t")
    out_dir = ensure_dir(Path("plots"))
    fig.tight_layout()
    fig.savefig(out_dir / f"trajectory_episode_{episode_id}.png")
