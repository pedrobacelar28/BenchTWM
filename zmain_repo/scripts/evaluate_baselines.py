"""Evaluate an environment's declared reference policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.envs.factory import make_env
from src.envs.reference import make_reference_policy
from src.evaluation import evaluate_policy
from src.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Experiment JSON config with env_id and obs_backend")
    parser.add_argument("--output-dir", default="runs/baselines")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--policies", nargs="+", default=["random", "smart", "soc", "oracle"])
    args = parser.parse_args()
    config = load_json(args.config)
    output_root = Path(args.output_dir)
    summaries = {}
    for name in args.policies:
        env = make_env(config, obs_backend=config["obs_backend"])
        policy = make_reference_policy(env, name, seed=int(config.get("seed", 0)))
        env.close()
        policy_config = {**config, "algo": "baseline", "method_id": name}
        result = evaluate_policy(
            policy,
            config=policy_config,
            num_episodes=args.episodes,
            eval_seed_base=int(config.get("eval_seed_base", 100000)),
            output_dir=output_root / name,
            checkpoint_name=name,
            export_csv=True,
            show_progress=True,
        )
        summaries[name] = result["summary"]
    save_json(summaries, output_root / "baseline_summary.json")


if __name__ == "__main__":
    main()
