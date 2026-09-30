#!/usr/bin/env python3
"""Emit deterministic JAX fixtures for the PyTorch Dreamer fidelity tests.

This script is intentionally outside the BenchTWM runtime and is not a test. Run
it from an isolated environment with JAX installed and compare its JSON output with
``tests/fixtures/dreamer_jax_reference.json``.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def rms(x, scale, eps=1e-4):
    return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + eps) * scale


def silu(x):
    return jax.nn.silu(x)


def linear(x, index, output):
    weight = jnp.arange(x.shape[-1] * output, dtype=jnp.float32).reshape(x.shape[-1], output)
    weight = (weight + index + 1) / (100.0 + index)
    bias = jnp.arange(output, dtype=jnp.float32) / (50.0 + index)
    return x @ weight + bias


def block_linear(x, blocks, output, index):
    grouped = x.reshape(x.shape[0], blocks, x.shape[-1] // blocks)
    shape = (blocks, x.shape[-1] // blocks, output // blocks)
    weight = jnp.arange(int(jnp.prod(jnp.asarray(shape))), dtype=jnp.float32).reshape(shape)
    weight = (weight + index + 1) / (200.0 + index)
    bias = jnp.arange(output, dtype=jnp.float32) / (80.0 + index)
    return jnp.einsum("bki,kio->bko", grouped, weight).reshape(x.shape[0], output) + bias


def rssm_core_fixture():
    deter = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 10.0 - 0.5
    stoch = jax.nn.one_hot(jnp.asarray([[0, 2], [1, 3]]), 4).reshape(2, 8)
    action = jax.nn.one_hot(jnp.asarray([1, 3]), 5)
    parts = [
        silu(rms(linear(deter, 0, 3), jnp.asarray([1.0, 0.8, 1.2]))),
        silu(rms(linear(stoch, 1, 3), jnp.asarray([0.9, 1.1, 1.0]))),
        silu(rms(linear(action, 2, 3), jnp.asarray([1.2, 1.0, 0.7]))),
    ]
    groups = 4
    grouped_deter = deter.reshape(2, groups, 2)
    repeated = jnp.repeat(jnp.concatenate(parts, -1)[:, None], groups, axis=1)
    x = jnp.concatenate([grouped_deter, repeated], -1).reshape(2, -1)
    x = silu(rms(block_linear(x, groups, 8, 3), jnp.linspace(0.8, 1.2, 8)))
    gates = block_linear(x, groups, 24, 4).reshape(2, groups, 6)
    reset, candidate, update = jnp.split(gates, 3, -1)
    reset = jax.nn.sigmoid(reset.reshape(2, 8))
    candidate = jnp.tanh(reset * candidate.reshape(2, 8))
    update = jax.nn.sigmoid(update.reshape(2, 8) - 1.0)
    return update * candidate + (1.0 - update) * deter


def lambda_return(last, term, reward, value, bootstrap, discount, lam):
    returns = [bootstrap[:, -1]]
    live = (1 - term.astype(jnp.float32))[:, 1:] * discount
    cont = (1 - last.astype(jnp.float32))[:, 1:] * lam
    intermediate = reward[:, 1:] + (1 - cont) * live * bootstrap[:, 1:]
    for time in reversed(range(live.shape[1])):
        returns.append(intermediate[:, time] + live[:, time] * cont[:, time] * returns[-1])
    return jnp.stack(list(reversed(returns))[:-1], 1)


def main():
    values = jnp.asarray([-100.0, -1.0, 0.0, 1.0, 100.0], jnp.float32)
    last = jnp.asarray([[0, 0, 0, 1]], bool)
    term = jnp.asarray([[0, 0, 0, 1]], bool)
    reward = jnp.asarray([[0.0, 1.0, 2.0, 3.0]], jnp.float32)
    value = jnp.asarray([[0.5, 0.6, 0.7, 0.8]], jnp.float32)
    payload = {
        "jax_version": jax.__version__,
        "symlog": symlog(values).tolist(),
        "symexp_roundtrip": symexp(symlog(values)).tolist(),
        "rms": rms(
            jnp.asarray([[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]], jnp.float32),
            jnp.asarray([1.0, 0.8, 1.2], jnp.float32),
        ).tolist(),
        "lambda_return": lambda_return(last, term, reward, value, value, 0.99, 0.95).tolist(),
        "rssm_core": rssm_core_fixture().tolist(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
