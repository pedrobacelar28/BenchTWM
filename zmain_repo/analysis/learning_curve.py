"""Plot learning curves from processed files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from analysis.colors import agent_colors
from analysis.utils import find_best, get_reference_stats, smoothen_runs
from src.utils.io import ensure_dir
from src.utils.json_handling import get_sorted_dict
from src.utils.methods import method_id, train_mode


def confidence_interval(mean, stderr):
    return mean - stderr, mean + stderr


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python analysis/learning_curve.py <legend y/n> <key> <metric> <json files>")
        raise SystemExit(1)

    show_legend = sys.argv[1].lower() == "y"
    key_to_plot = sys.argv[2]
    metric = sys.argv[3]
    json_files = sys.argv[4:]
    fig, ax = plt.subplots(1, figsize=(7, 4), dpi=200)

    for json_file in json_files:
        config = get_sorted_dict(json_file)
        _, params, payload = find_best(config, data=key_to_plot, metric=metric, minmax="max")
        if payload is None or params is None:
            continue
        mean = smoothen_runs(payload[key_to_plot]["mean"])
        stderr = payload[key_to_plot]["stderr"]
        if train_mode(params) == "offline" and "offline_updates" in payload:
            x_values = payload["offline_updates"]["mean"][: len(mean)]
        else:
            x_values = range(len(mean))
        label = method_id(params)
        color = agent_colors.get(label, agent_colors.get(params["algo"], None))
        ax.plot(x_values, mean, label=label, color=color, linewidth=1.5)
        low, high = confidence_interval(mean, stderr[: len(mean)])
        ax.fill_between(x_values, low, high, color=color, alpha=0.2)

    refs = get_reference_stats()
    ax.axhline(refs["expert_policy"]["j_mean"], linestyle="--", linewidth=1.0, color=agent_colors["expert"], label="expert")
    ax.axhline(refs["pi_star"]["j_mean"], linestyle=":", linewidth=1.0, color=agent_colors["optimal"], label="optimal")
    if json_files and train_mode(get_sorted_dict(json_files[0])) == "offline":
        ax.set_xlabel("Offline Gradient Updates")
    else:
        ax.set_xlabel("Episodes")
    ax.set_ylabel(key_to_plot.replace("_", " ").title())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend()
    out_dir = ensure_dir(Path("plots"))
    fig.tight_layout()
    fig.savefig(out_dir / f"learning_curve_{key_to_plot}.png")
