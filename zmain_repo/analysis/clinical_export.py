"""Re-export trajectories from a saved checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.append(os.getcwd())

from src.algorithms.registry import get_policy_loader
from src.evaluation import evaluate_policy
from src.utils.io import load_json
from src.utils.torch_utils import get_torch_device


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analysis/clinical_export.py <run_dir> <best|last>")
        raise SystemExit(1)
    run_dir = Path(sys.argv[1])
    checkpoint_name = sys.argv[2]
    config = load_json(run_dir / "config.json")
    checkpoint = torch.load(run_dir / f"checkpoint_{checkpoint_name}.pt", map_location=get_torch_device())
    policy = get_policy_loader(config["algo"])(config, checkpoint)
    evaluate_policy(
        policy,
        config=config,
        num_episodes=int(config["eval_episodes"]),
        eval_seed_base=int(config["eval_seed_base"]),
        output_dir=run_dir,
        checkpoint_name=checkpoint_name,
        export_csv=True,
    )
