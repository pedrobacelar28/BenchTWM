"""Environment selection from experiment configuration."""

from __future__ import annotations

from typing import Any

ENV_ALIASES = {
    "icu": "icu_sepsis",
    "icu_sepsis": "icu_sepsis",
    "epicare": "epicare",
}


def environment_id(config: dict | None) -> str:
    raw = "icu_sepsis" if config is None else str(config.get("env_id", "icu_sepsis")).lower()
    try:
        return ENV_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(f"Unknown env_id={raw!r}; expected one of {sorted(set(ENV_ALIASES.values()))}") from exc


def make_env(
    config: dict | None = None,
    *,
    obs_backend: str | None = None,
    seed: int | None = None,
    params: Any = None,
):
    config = {} if config is None else config
    env_id = environment_id(config)
    backend = str(obs_backend or config.get("obs_backend", "discrete_id"))
    kwargs = dict(config.get("env_kwargs", {}))
    environment_seed = int(config.get("environment_seed", config.get("seed", 0)))
    if env_id == "icu_sepsis":
        from src.envs.icu_adapter import ICUSepsisEnvWrapper

        return ICUSepsisEnvWrapper(
            obs_backend=backend,
            seed=seed,
            params=params if params is not None else kwargs or None,
        )
    from src.envs.epicare_adapter import EpiCareEnvWrapper

    return EpiCareEnvWrapper(
        obs_backend=backend,
        seed=seed,
        environment_seed=environment_seed,
        **kwargs,
    )
