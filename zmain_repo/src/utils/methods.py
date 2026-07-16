"""Method naming helpers."""

from __future__ import annotations


OFFLINE_METHOD_IDS = {
    "dqn": "dqn_offline",
    "ppo": "ppo_logged_offline",
}

CEM_MODEL_ALGOS = {"raw_model", "latent_model", "reconstructive_model"}
BACKEND_LABELS = {
    "discrete_id": "discrete",
    "state_vector": "continuous",
}


def train_mode(config: dict) -> str:
    return str(config.get("train_mode", "online")).lower()


def _cem_method_id(config: dict) -> str:
    obs_backend = str(config.get("obs_backend", ""))
    if obs_backend not in BACKEND_LABELS:
        raise KeyError(f"CEM world-model algo requires obs_backend in {sorted(BACKEND_LABELS)}, got {obs_backend!r}")
    base = f"{config['algo']}_{BACKEND_LABELS[obs_backend]}_cem"
    if train_mode(config) == "offline":
        return f"{base}_offline"
    return base


def method_id(config: dict) -> str:
    if "method_id" in config:
        return str(config["method_id"])
    algo = str(config["algo"])
    if algo in CEM_MODEL_ALGOS:
        return _cem_method_id(config)
    if train_mode(config) == "offline":
        if algo not in OFFLINE_METHOD_IDS:
            raise KeyError(f"No offline method_id registered for algo={algo!r}")
        return OFFLINE_METHOD_IDS[algo]
    return algo
