"""Generate local offline ICU-Sepsis datasets from simulator rollouts."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import trange

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.algorithms.registry import get_policy_loader
from src.bootstrap import bootstrap_icu_helpers
from src.envs.icu_adapter import ICUSepsisEnvWrapper
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.json_handling import get_param_iterable
from src.utils.metrics import action_stability


POLICY_IDS = {
    "random_admissible": 0,
    "expert": 1,
    "optimal": 2,
    "epsilon_expert": 3,
    "policy_checkpoint": 4,
}


def _resolve_output_dir(config: dict) -> Path:
    if "output_dir" in config:
        return Path(config["output_dir"])
    if "dataset_path" in config:
        return Path(config["dataset_path"]).parent
    raise KeyError("Dataset config must define output_dir or dataset_path")


def _normalise_probs(probs: np.ndarray, admissible_actions: list[int]) -> np.ndarray:
    masked = np.zeros_like(probs, dtype=np.float64)
    masked[np.asarray(admissible_actions, dtype=np.int64)] = probs[np.asarray(admissible_actions, dtype=np.int64)]
    total = float(masked.sum())
    if total <= 0.0:
        masked[np.asarray(admissible_actions, dtype=np.int64)] = 1.0 / max(1, len(admissible_actions))
        return masked
    return masked / total


def _uniform_admissible_probs(num_actions: int, admissible_actions: list[int]) -> np.ndarray:
    probs = np.zeros(num_actions, dtype=np.float64)
    probs[np.asarray(admissible_actions, dtype=np.int64)] = 1.0 / max(1, len(admissible_actions))
    return probs


def _sample_from_probs(rng: np.random.Generator, probs: np.ndarray) -> tuple[int, float]:
    action = int(rng.choice(len(probs), p=probs))
    return action, float(math.log(max(probs[action], 1e-12)))


def _build_optimal_policy(env: ICUSepsisEnvWrapper, config: dict) -> np.ndarray:
    bootstrap_icu_helpers()
    from icu_sepsis_helpers.utils.mdp import value_iteration

    dynamics = env.env.unwrapped.dynamics
    pi_star, _ = value_iteration(
        dynamics["tx_mat"],
        dynamics["r_mat"],
        float(config.get("gamma", 1.0)),
        max_steps=int(config.get("optimal_value_iteration_steps", 50000)),
        delta=float(config.get("optimal_value_iteration_delta", 1e-6)),
    )
    return np.asarray(pi_star, dtype=np.int64)


def _load_checkpoint_policy(config: dict):
    if config.get("behavior_policy") != "policy_checkpoint" and "policy_checkpoint" not in config.get("mixed_weights", {}):
        return None
    policy_config_raw = load_json(config["policy_config_path"])
    policy_index = int(config.get("policy_config_index", 0))
    policy_configs = get_param_iterable(policy_config_raw)
    policy_config = policy_configs[policy_index % len(policy_configs)]
    import torch

    checkpoint = torch.load(config["policy_checkpoint_path"], map_location="cpu")
    return get_policy_loader(policy_config["algo"])(policy_config, checkpoint)


class BehaviorPolicy:
    def __init__(self, config: dict, env: ICUSepsisEnvWrapper):
        self.config = config
        self.rng = np.random.default_rng(int(config["seed"]))
        self.num_actions = env.num_actions
        self.expert_policy = env.env.unwrapped.expert_policy
        self.optimal_policy = None
        self.checkpoint_policy = _load_checkpoint_policy(config)
        policies = {str(config["behavior_policy"])}
        policies.update(str(key) for key in config.get("mixed_weights", {}).keys())
        if "optimal" in policies:
            self.optimal_policy = _build_optimal_policy(env, config)

    def choose_episode_policy(self) -> str:
        policy = str(self.config["behavior_policy"])
        if policy != "mixed":
            return policy
        weights = self.config.get("mixed_weights", {"random_admissible": 0.34, "expert": 0.33, "optimal": 0.33})
        names = list(weights.keys())
        probs = np.asarray([weights[name] for name in names], dtype=np.float64)
        probs = probs / probs.sum()
        return str(self.rng.choice(names, p=probs))

    def action(self, episode_policy: str, obs, info: dict) -> tuple[int, float, int]:
        admissible = list(info["admissible_actions"])
        state_id = int(info["state_id"])
        if episode_policy == "random_admissible":
            probs = _uniform_admissible_probs(self.num_actions, admissible)
            action, logprob = _sample_from_probs(self.rng, probs)
        elif episode_policy == "expert":
            probs = _normalise_probs(self.expert_policy[state_id], admissible)
            action, logprob = _sample_from_probs(self.rng, probs)
        elif episode_policy == "optimal":
            if self.optimal_policy is None:
                raise RuntimeError("optimal policy was not initialized")
            action = int(self.optimal_policy[state_id])
            if action not in admissible:
                action = int(admissible[0])
            logprob = 0.0
        elif episode_policy == "epsilon_expert":
            epsilon = float(self.config.get("epsilon", 0.2))
            expert_probs = _normalise_probs(self.expert_policy[state_id], admissible)
            random_probs = _uniform_admissible_probs(self.num_actions, admissible)
            probs = (1.0 - epsilon) * expert_probs + epsilon * random_probs
            probs = probs / probs.sum()
            action, logprob = _sample_from_probs(self.rng, probs)
        elif episode_policy == "policy_checkpoint":
            if self.checkpoint_policy is None:
                raise RuntimeError("policy_checkpoint requires policy_config_path and policy_checkpoint_path")
            action = int(self.checkpoint_policy.act(obs, info, deterministic=True))
            logprob = 0.0
        else:
            raise ValueError(f"Unsupported behavior_policy={episode_policy!r}")
        return action, logprob, POLICY_IDS[episode_policy]


def _empty_buffers() -> dict[str, list]:
    return {
        "observations": [],
        "next_observations": [],
        "state_vectors": [],
        "next_state_vectors": [],
        "actions": [],
        "next_actions": [],
        "rewards": [],
        "terminals": [],
        "state_ids": [],
        "next_state_ids": [],
        "action_masks": [],
        "next_action_masks": [],
        "behavior_logprobs": [],
        "behavior_policy_ids": [],
        "episode_ids": [],
        "timesteps": [],
        "sofa_scores": [],
        "next_sofa_scores": [],
    }


def _append_episode(buffers: dict[str, list], episode_records: list[dict]) -> None:
    actions = [record["actions"] for record in episode_records]
    for idx, record in enumerate(episode_records):
        record = dict(record)
        record["next_actions"] = actions[idx + 1] if idx + 1 < len(actions) else -1
        for key, value in record.items():
            buffers[key].append(value)


def _collect_split(
    *,
    config: dict,
    env: ICUSepsisEnvWrapper,
    behavior: BehaviorPolicy,
    num_episodes: int,
    start_episode_id: int,
    seed_offset: int,
) -> tuple[dict[str, list], list[dict]]:
    buffers = _empty_buffers()
    summaries: list[dict] = []
    for local_episode in trange(num_episodes, desc=f"collect_{seed_offset}", dynamic_ncols=True):
        episode_id = start_episode_id + local_episode
        obs, info = env.reset(seed=int(config["seed"]) + seed_offset + local_episode)
        episode_policy = behavior.choose_episode_policy()
        done = False
        total_reward = 0.0
        t = 0
        episode_records: list[dict] = []
        episode_actions: list[int] = []
        sofas = [float(info["sofa_score"])]
        initial_state = int(info["state_id"])
        initial_sofa = float(info["sofa_score"])
        while not done:
            action, logprob, policy_id = behavior.action(episode_policy, obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            episode_records.append(
                {
                    "observations": np.asarray(obs, dtype=np.float32),
                    "next_observations": np.asarray(next_obs, dtype=np.float32),
                    "state_vectors": np.asarray(info["state_vector"], dtype=np.float32),
                    "next_state_vectors": np.asarray(next_info["state_vector"], dtype=np.float32),
                    "actions": int(action),
                    "rewards": float(reward),
                    "terminals": bool(done),
                    "state_ids": int(info["state_id"]),
                    "next_state_ids": int(next_info["state_id"]),
                    "action_masks": np.asarray(info["action_mask"], dtype=np.float32),
                    "next_action_masks": np.asarray(next_info["action_mask"], dtype=np.float32),
                    "behavior_logprobs": float(logprob),
                    "behavior_policy_ids": int(policy_id),
                    "episode_ids": int(episode_id),
                    "timesteps": int(t),
                    "sofa_scores": float(info["sofa_score"]),
                    "next_sofa_scores": float(next_info["sofa_score"]),
                }
            )
            episode_actions.append(action)
            total_reward += float(reward)
            obs = next_obs
            info = next_info
            sofas.append(float(info["sofa_score"]))
            t += 1
        _append_episode(buffers, episode_records)
        stability_l1, abrupt_rate = action_stability(episode_actions)
        summaries.append(
            {
                "episode_id": episode_id,
                "behavior_policy": episode_policy,
                "return": float(total_reward),
                "survived": bool(total_reward > 0),
                "length": int(t),
                "initial_state": initial_state,
                "initial_sofa": initial_sofa,
                "mean_sofa": float(sum(sofas) / len(sofas)),
                "action_stability_l1": stability_l1,
                "abrupt_jump_rate": abrupt_rate,
            }
        )
    return buffers, summaries


def _stack_buffers(buffers: dict[str, list]) -> dict[str, np.ndarray]:
    arrays = {}
    for key, values in buffers.items():
        if key in {"observations", "next_observations", "action_masks", "next_action_masks"}:
            arrays[key] = np.asarray(values, dtype=np.uint8)
        elif key in {"state_vectors", "next_state_vectors"}:
            arrays[key] = np.asarray(values, dtype=np.float32)
        elif key in {"actions", "next_actions", "state_ids", "next_state_ids", "behavior_policy_ids", "episode_ids", "timesteps"}:
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key == "terminals":
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    return arrays


def _save_split(path: Path, buffers: dict[str, list]) -> dict:
    ensure_dir(path.parent)
    arrays = _stack_buffers(buffers)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "num_transitions": int(len(arrays["actions"])),
        "num_episodes": int(len(np.unique(arrays["episode_ids"]))),
    }


def _summary_stats(summaries: list[dict]) -> dict:
    df = pd.DataFrame(summaries)
    policy_counts = Counter(df["behavior_policy"].tolist()) if not df.empty else Counter()
    return {
        "num_episodes": int(len(df)),
        "num_transitions": int(df["length"].sum()) if not df.empty else 0,
        "return_mean": float(df["return"].mean()) if not df.empty else 0.0,
        "survival_rate": float(df["survived"].mean()) if not df.empty else 0.0,
        "mean_episode_length": float(df["length"].mean()) if not df.empty else 0.0,
        "behavior_policy_counts": dict(policy_counts),
    }


def generate_dataset(config: dict) -> dict:
    output_dir = _resolve_output_dir(config)
    output_dir = output_dir if output_dir.is_absolute() else Path.cwd() / output_dir
    train_path = Path(config.get("dataset_path", output_dir / "train.npz"))
    val_path = Path(config.get("val_dataset_path", output_dir / "val.npz"))
    if not train_path.is_absolute():
        train_path = Path.cwd() / train_path
    if not val_path.is_absolute():
        val_path = Path.cwd() / val_path
    metadata_path = output_dir / "metadata.json"
    episodes_path = output_dir / "episodes_summary.csv"
    if train_path.exists() and val_path.exists() and metadata_path.exists() and not bool(config.get("overwrite", False)):
        print(f"Dataset already exists: {output_dir}")
        return load_json(metadata_path)

    env = ICUSepsisEnvWrapper(obs_backend="discrete_id", seed=int(config["seed"]))
    behavior = BehaviorPolicy(config, env)
    train_buffers, train_summaries = _collect_split(
        config=config,
        env=env,
        behavior=behavior,
        num_episodes=int(config["num_train_episodes"]),
        start_episode_id=0,
        seed_offset=0,
    )
    val_buffers, val_summaries = _collect_split(
        config=config,
        env=env,
        behavior=behavior,
        num_episodes=int(config["num_val_episodes"]),
        start_episode_id=int(config["num_train_episodes"]),
        seed_offset=int(config["num_train_episodes"]) + 100000,
    )
    env.close()

    train_meta = _save_split(train_path, train_buffers)
    val_meta = _save_split(val_path, val_buffers)
    all_summaries = train_summaries + val_summaries
    ensure_dir(output_dir)
    pd.DataFrame(all_summaries).to_csv(episodes_path, index=False)
    metadata = {
        "config": config,
        "policy_ids": POLICY_IDS,
        "train": {**train_meta, **_summary_stats(train_summaries)},
        "validation": {**val_meta, **_summary_stats(val_summaries)},
        "episodes_summary_path": str(episodes_path),
    }
    save_json(metadata, metadata_path)
    print(f"Saved dataset under {output_dir}")
    return metadata


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/offline/generate_dataset.py <dataset_config.json>")
        raise SystemExit(1)
    cfg = load_json(sys.argv[1])
    generate_dataset(cfg)
