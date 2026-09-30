"""Environment-agnostic policy evaluation and trajectory exports."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from benchtwm.utils.progress import tqdm

from benchtwm.envs.factory import make_env
from benchtwm.utils.io import ensure_dir, save_json
from benchtwm.utils.metrics import action_stability
from benchtwm.utils.methods import method_id
from benchtwm.utils.queue_progress import update_progress


def _metric_summary(episodes: pd.DataFrame, name: str) -> float:
    column = f"mean_{name}"
    return float(episodes[column].mean()) if column in episodes and not episodes.empty else 0.0


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
    deterministic: bool | None = None,
) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else None
    deterministic = bool(config.get("eval_deterministic", True) if deterministic is None else deterministic)
    env = make_env(config, obs_backend=config["obs_backend"], params=params)
    if hasattr(policy, "bind_env"):
        policy.bind_env(env)
    method = method_id(config)
    trajectories: list[dict] = []
    episodes: list[dict] = []
    action_latencies: list[float] = []
    planner_calls = 0
    planner_model_evaluations = 0
    invalid_action_count = 0
    episode_iter = tqdm(
        range(num_episodes),
        desc=progress_desc or f"eval:{checkpoint_name}",
        unit="ep",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    progress_every = max(1, int(num_episodes) // 20)
    update_progress(
        status="running",
        phase="evaluation",
        method_id=config.get("method_id"),
        step=0,
        total_steps=int(num_episodes),
    )
    for episode_id in episode_iter:
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()
        obs, info = env.reset(seed=eval_seed_base + episode_id)
        done = False
        total_reward = 0.0
        step_idx = 0
        actions: list[int] = []
        metric_series = [env.transition_metrics(info)]
        while not done:
            action_started = time.perf_counter()
            action = int(policy.act(obs, info, deterministic=deterministic))
            action_latencies.append(time.perf_counter() - action_started)
            planner = getattr(policy, "planner", None)
            if planner is not None:
                planner_calls += 1
                planner_model_evaluations += int(
                    getattr(planner, "samples", 0)
                    * getattr(planner, "iterations", 0)
                    * getattr(planner, "horizon", 0)
                )
            if action not in set(int(item) for item in info.get("admissible_actions", [])):
                invalid_action_count += 1
                raise ValueError(
                    f"Policy {method!r} emitted inadmissible action {action}; "
                    f"admissible={info.get('admissible_actions', [])}"
                )
            current_metrics = env.transition_metrics(info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            if hasattr(policy, "observe"):
                policy.observe(action, reward)
            done = bool(terminated or truncated)
            row = {
                "env_id": env.spec.env_id,
                "method": method,
                "algo": config["algo"],
                "method_id": method,
                "seed": int(config["seed"]),
                "eval_seed": int(eval_seed_base + episode_id),
                "checkpoint": checkpoint_name,
                "episode_id": episode_id,
                "t": step_idx,
                "state_id": int(info["state_id"]),
                "action_idx": action,
                "action_label": env.action_label(action),
                "reward": float(reward),
                "done": done,
                "success": False,
                **{f"metric_{name}": float(value) for name, value in current_metrics.items()},
                **{
                    f"feature_{name}": float(value)
                    for name, value in zip(env.spec.feature_names, np.asarray(info["state_vector"], dtype=np.float32))
                },
            }
            if env.spec.env_id == "icu_sepsis":
                fluid_bin, vaso_bin = env.decode_action(action)
                row.update(
                    {
                        "sofa_score": float(current_metrics["sofa_score"]),
                        "fluid_bin": int(fluid_bin),
                        "vaso_bin": int(vaso_bin),
                        "survived": False,
                    }
                )
                if config["obs_backend"] == "state_vector":
                    row.update({f"state_vector_{idx}": float(value) for idx, value in enumerate(info["state_vector"])})
            trajectories.append(row)
            actions.append(action)
            total_reward += float(reward)
            obs = next_obs
            info = next_info
            metric_series.append(env.transition_metrics(info))
            step_idx += 1

        outcome = env.episode_metrics(total_reward=total_reward, final_info=info, transition_metrics=metric_series)
        stability_l1, abrupt_rate = action_stability(
            actions,
            env.action_distance,
            abrupt_threshold=0.5,
            abrupt_fn=env.action_is_abrupt,
        )
        episode = {
            "env_id": env.spec.env_id,
            "method": method,
            "algo": config["algo"],
            "method_id": method,
            "seed": int(config["seed"]),
            "eval_seed": int(eval_seed_base + episode_id),
            "checkpoint": checkpoint_name,
            "episode_id": episode_id,
            "return": float(total_reward),
            "success": bool(outcome["success"]),
            "length": int(step_idx),
            "initial_state": int(metric_series and trajectories[-step_idx]["state_id"]),
            "action_stability_l1": stability_l1,
            "abrupt_jump_rate": abrupt_rate,
            **outcome,
        }
        for metric_name in env.spec.metric_names:
            values = [row[metric_name] for row in metric_series]
            episode[f"initial_{metric_name}"] = float(values[0])
            episode[f"mean_{metric_name}"] = float(np.mean(values))
            episode[f"final_{metric_name}"] = float(values[-1])
        if env.spec.env_id == "icu_sepsis":
            episode.update(
                {
                    "survived": bool(outcome["success"]),
                    "initial_sofa": episode["initial_sofa_score"],
                    "mean_sofa": episode["mean_sofa_score"],
                }
            )
        episodes.append(episode)
        for row in trajectories[-step_idx:]:
            row["success"] = bool(outcome["success"])
            if env.spec.env_id == "icu_sepsis":
                row["survived"] = bool(outcome["success"])
        if show_progress:
            episode_iter.set_postfix(ret=f"{total_reward:.3f}", len=step_idx, success=int(outcome["success"]))
        if episode_id == 0 or (episode_id + 1) % progress_every == 0 or episode_id + 1 == int(num_episodes):
            update_progress(
                status="running",
                phase="evaluation",
                method_id=config.get("method_id"),
                step=episode_id + 1,
                total_steps=int(num_episodes),
            )

    env.close()
    episodes_df = pd.DataFrame(episodes)
    trajectories_df = pd.DataFrame(trajectories)
    summary = {
        "mean_return": float(episodes_df["return"].mean()) if not episodes_df.empty else 0.0,
        "mean_success": float(episodes_df["success"].mean()) if not episodes_df.empty else 0.0,
        "mean_length": float(episodes_df["length"].mean()) if not episodes_df.empty else 0.0,
        "mean_action_stability_l1": float(episodes_df["action_stability_l1"].mean()) if not episodes_df.empty else 0.0,
        "mean_abrupt_jump_rate": float(episodes_df["abrupt_jump_rate"].mean()) if not episodes_df.empty else 0.0,
        "mean_action_latency_ms": float(np.mean(action_latencies) * 1000.0) if action_latencies else 0.0,
        "planner_calls": int(planner_calls),
        "planner_model_evaluations": int(planner_model_evaluations),
        "invalid_action_count": int(invalid_action_count),
        "invalid_action_rate": float(invalid_action_count / max(1, len(trajectories))),
        **{f"mean_{name}": _metric_summary(episodes_df, name) for name in env.spec.metric_names},
    }
    if "remission" in episodes_df:
        summary["mean_remission_rate"] = float(episodes_df["remission"].mean()) if not episodes_df.empty else 0.0
    if "adverse_event" in episodes_df:
        summary["mean_adverse_event_rate"] = float(episodes_df["adverse_event"].mean()) if not episodes_df.empty else 0.0
    if "time_to_remission" in episodes_df:
        observed = episodes_df["time_to_remission"].dropna()
        summary["mean_time_to_remission"] = float(observed.mean()) if not observed.empty else 0.0
    # Retained as an alias for existing ICU-oriented trainers and artifacts.
    summary["mean_survival"] = summary["mean_success"]
    if output_dir is not None:
        ensure_dir(output_dir)
        save_json(env.spec.to_dict(), output_dir / "environment_spec.json")
        if export_csv:
            episodes_df.to_csv(output_dir / "episodes_summary.csv", index=False)
            trajectories_df.to_csv(output_dir / "trajectories.csv", index=False)
    return {"episodes": episodes_df, "trajectories": trajectories_df, "summary": summary, "environment": env.spec.to_dict()}
