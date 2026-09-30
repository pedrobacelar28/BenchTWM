"""Environment wrappers and their shared factory."""


def make_env(*args, **kwargs):
    from benchtwm.envs.factory import make_env as _make_env

    return _make_env(*args, **kwargs)


__all__ = ["make_env"]
