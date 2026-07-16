"""Checkpoint evaluation, clinical export, and policy rollouts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.envs.icu_adapter import ICUSepsisEnvWrapper
from src.utils.io import ensure_dir
from src.utils.metrics import action_stability
from src.utils.methods import method_id


def evaluate_policy(
    policy,
    *,
    config: dict,
    num_episodes: int,
    eval_seed_base: int,
    output_dir: str | Path | None = None,
    checkpoint_name: str = "best",
    params=None,
    export_csv: bool = True,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else None
    env = ICUSepsisEnvWrapper(obs_backend=config["obs_backend"], params=params)
    method = method_id(config)
    trajectories: list[dict] = []
    episodes: list[dict] = []

    episode_iter = tqdm(
        range(num_episodes),
        desc=progress_desc or f"eval:{checkpoint_name}",
        unit="ep",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress or not sys.stderr.isatty(),
    )
    for episode_id in episode_iter:
        obs, info = env.reset(seed=eval_seed_base + episode_id)
        done = False
        total_reward = 0.0
        step_idx = 0
        actions = []
        sofas = [float(info["sofa_score"])]
        initial_state = int(info["state_id"])
        initial_sofa = float(info["sofa_score"])
        while not done:
            action = int(policy.act(obs, info, deterministic=True))
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            actions.append(action)
            trajectories.append(
                {
                    "method": method,
                    "algo": config["algo"],
                    "method_id": method,
                    "seed": int(config["seed"]),
                    "eval_seed": int(eval_seed_base + episode_id),
                    "checkpoint": checkpoint_name,
                    "episode_id": episode_id,
                    "t": step_idx,
                    "state_id": int(info["state_id"]),
                    "sofa_score": float(info["sofa_score"]),
                    "action_idx": int(action),
                    "fluid_bin": int(env.decode_action(action)[0]),
                    "vaso_bin": int(env.decode_action(action)[1]),
                    "reward": float(reward),
                    "done": bool(done),
                    "survived": False,
                    **(
                        {f"state_vector_{i}": float(value) for i, value in enumerate(info["state_vector"])}
                        if config["obs_backend"] == "state_vector"
                        else {}
                    ),
                }
            )
            total_reward += float(reward)
            obs = next_obs
            info = next_info
            sofas.append(float(info["sofa_score"]))
            step_idx += 1

        survived = bool(total_reward > 0)
        stability_l1, abrupt_rate = action_stability(actions)
        episodes.append(
            {
                "method": method,
                "algo": config["algo"],
                "method_id": method,
                "seed": int(config["seed"]),
                "eval_seed": int(eval_seed_base + episode_id),
                "checkpoint": checkpoint_name,
                "episode_id": episode_id,
                "return": float(total_reward),
                "survived": survived,
                "length": int(step_idx),
                "initial_state": initial_state,
                "initial_sofa": initial_sofa,
                "mean_sofa": float(sum(sofas) / len(sofas)),
                "action_stability_l1": stability_l1,
                "abrupt_jump_rate": abrupt_rate,
            }
        )
        for row in trajectories[-step_idx:]:
            row["survived"] = survived
        if show_progress:
            episode_iter.set_postfix(
                ret=f"{total_reward:.3f}",
                len=step_idx,
                surv=int(survived),
            )

    env.close()
    episodes_df = pd.DataFrame(episodes)
    trajectories_df = pd.DataFrame(trajectories)
    if export_csv and output_dir is not None:
        ensure_dir(output_dir)
        episodes_df.to_csv(output_dir / "episodes_summary.csv", index=False)
        trajectories_df.to_csv(output_dir / "trajectories.csv", index=False)
    return {
        "episodes": episodes_df,
        "trajectories": trajectories_df,
        "summary": {
            "mean_return": float(episodes_df["return"].mean()) if not episodes_df.empty else 0.0,
            "mean_survival": float(episodes_df["survived"].mean()) if not episodes_df.empty else 0.0,
            "mean_length": float(episodes_df["length"].mean()) if not episodes_df.empty else 0.0,
            "mean_action_stability_l1": float(episodes_df["action_stability_l1"].mean()) if not episodes_df.empty else 0.0,
            "mean_abrupt_jump_rate": float(episodes_df["abrupt_jump_rate"].mean()) if not episodes_df.empty else 0.0,
        },
    }
