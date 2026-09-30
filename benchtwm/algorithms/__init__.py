"""Offline RL algorithms and world-model agents."""


def load_policy(config: dict, checkpoint: dict):
    """Rebuild the deployed policy of a trained run from its config and checkpoint."""

    from benchtwm.algorithms.registry import get_policy_loader

    return get_policy_loader(config["algo"])(config, checkpoint)
