"""Method naming helpers."""

from __future__ import annotations


OFFLINE_METHOD_IDS = {
    "bc": "bc_offline",
    "dqn": "dqn_offline",
    "iql": "iql_offline",
    "cql": "cql_offline",
    "ppo": "ppo_logged_offline",
}

CEM_MODEL_ALGOS = {"raw_model", "latent_model", "reconstructive_model"}
BACKEND_LABELS = {
    "discrete_id": "discrete",
    "state_vector": "continuous",
}

TRACKS = {"direct_rl", "provisional_planning", "published_planning", "online_baseline"}


def action_type(config: dict) -> str:
    """Return the public action-space label used by benchmark task IDs."""

    value = str(config.get("action_type", "discrete")).lower()
    if value != "discrete":
        raise ValueError(f"BenchTWM v0 supports action_type='discrete', got {value!r}")
    return value


def task_id(config: dict, dataset_id: str | None = None) -> str:
    """Build a stable task identity without making tasks path-specific."""

    dataset = str(dataset_id or config.get("dataset_id", "adhoc"))
    backend = str(config.get("obs_backend", "state_vector"))
    mode = train_mode(config)
    return f"{dataset}|obs={backend}|action={action_type(config)}|mode={mode}"


def benchmark_track(config: dict) -> str:
    """Resolve the experiment track used by reports and the v0 leaderboard."""

    explicit = config.get("track")
    if explicit not in (None, "", False):
        value = str(explicit)
        if value not in TRACKS:
            raise ValueError(f"Unsupported benchmark track={value!r}; expected one of {sorted(TRACKS)}")
        return value
    algo = str(config.get("algo", ""))
    if algo in CEM_MODEL_ALGOS:
        return "provisional_planning"
    if train_mode(config) == "offline" and algo in {"bc", "dqn", "iql", "cql"}:
        return "direct_rl"
    return "online_baseline"


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
