# DreamerV3 PyTorch Fidelity Audit

## Authority and scope

The authority is commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe` of
`danijar/dreamerv3`, audited from `repos_usados/dreamerv3/`. BenchTWM ports the
complete path exercised by structured vector observations and categorical
actions. Image CNNs, continuous-action distributions, online collection, and
the distributed JAX runtime are outside this benchmark contract; they are not
silently replaced by approximations.

The method is called **DreamerV3-style offline adaptation** because the replay
source is a fixed NPZ dataset and the framework is PyTorch. Within the selected
agent path, module order, stop-gradient placement, targets, losses, and update
rules are required to match the JAX source.

## Module correspondence

| JAX source | PyTorch port (`benchtwm/algorithms/dreamer/agent.py`) | Fidelity requirement |
|---|---|---|
| `rssm.Encoder` vector branch | `VectorEncoder` | symlog, Linear, RMSNorm, SiLU ordering |
| `rssm.RSSM` | `RSSM` | blocked recurrent core, posterior/prior, unimix one-hot sampling, balanced KL and free nats |
| `rssm.Decoder` vector branch | `VectorDecoder` | latent concatenation, MLP, symlog-MSE reconstruction |
| reward/value `MLPHead` | `reward_head`, `value_head` | symmetric 255-bin two-hot outputs |
| continuation `MLPHead` | `continue_head` | binary log-likelihood and horizon-adjusted target |
| categorical policy head | `policy_head` | categorical sampling, log probability, entropy |
| `SlowModel` | `slow_value_head` | source update order and rate 0.02 |
| `Normalize` | `RunningNormalize` | percentile, mean/std, none, limits, rates, and debias behavior |
| `imag_loss` | `DreamerAgent.loss` imagination section | continuation weights, lambda-return, return/advantage normalization, policy/value losses |
| `repl_loss` | `DreamerAgent.loss` replay-value section | replay lambda-return, bootstrap, slow regularization |
| Optax chain | `DreamerOptimizer` | per-tensor AGC, bias-corrected RMS, bias-corrected momentum, zero-start warmup |

## BenchTWM boundary adaptations

- The offline loader reconstructs Dreamer's reward-on-arrival record alignment
  and preserves carry across consecutive chunks. It does not change model
  equations.
- `action_mask` is applied only to deployed actor logits. It is not an
  observation feature and does not add a loss to the Dreamer core.
- The observation is always the environment's `float32` vector. Internal
  categorical stochastic states and one-hot action embeddings are not agent
  observations.
- `configs/algorithms/dreamer.json` uses bfloat16 compute on CUDA, matching
  the source JAX configuration. CPU execution and tiny tests fall back to float32 to avoid
  unsupported kernels; parity tolerances account for this device boundary.
- The current dataset schema combines environment termination and horizon
  truncation in `terminals`. Dreamer therefore receives that recorded signal
  for both `is_last` and `is_terminal`; no hidden simulator state is consulted.

## Verification gate

`tests/test_dreamer_fidelity.py` independently checks source formulas for
symlog/symexp, RMS normalization, categorical KL/unimix, symmetric two-hot
prediction/loss, blocked RSSM recurrence, lambda-return, slow-value updates,
and the AGC/RMS/momentum optimizer order. `tests/test_dreamer.py` checks
sequence boundaries and carry, the recurrent policy lifecycle, action masks,
and checkpoint round trips; the contract tests in `tests/test_algorithms.py`
and `tests/test_ope.py` cover training, deployment, and OPE on every
environment.

The pinned fixture in `tests/fixtures/dreamer_jax_reference.json` was generated
with JAX 0.4.33 by `tests/fixtures/dreamer_jax_reference.py`. It compares fixed
symlog, RMS, lambda-return, and blocked RSSM-core outputs against the PyTorch
modules with identical tensors and parameters.

The production dependency set intentionally does not install JAX. A
cross-framework run may be performed in an isolated reference environment,
but JAX, Ninjax, Embodied, and accelerator-specific packages must not become
BenchTWM runtime dependencies.
