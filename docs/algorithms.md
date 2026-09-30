# Algorithms

| Method id | Family | Summary |
|---|---|---|
| `bc` | model-free | Masked cross-entropy on logged actions |
| `dqn` | model-free | Offline DQN / fitted Q-iteration with a soft-updated target network |
| `iql` | model-free | Discrete IQL: expectile value, twin critics, advantage-weighted actor |
| `cql` | model-free | Discrete CQL: TD regression plus a conservative logsumexp penalty |
| `raw-model` | model-based | Observation-space dynamics model + CEM planning |
| `latent-model` | model-based | Latent dynamics against an EMA target encoder + CEM planning |
| `reconstructive-model` | model-based | Latent dynamics with next-observation reconstruction + CEM planning |
| `dreamer` | model-based | PyTorch port of DreamerV3 for vector observations, imagined actor-critic |

All methods read `float32` observation vectors and respect the action mask:
Q-values and logits of inadmissible actions are set to `-1e9`, and the CEM
planner gives inadmissible first actions exactly zero probability.

## The algorithm interface

Each method is one module in `benchtwm/algorithms/` defining an
`OfflineAlgorithm` subclass (`benchtwm/algorithms/base.py`):

```python
@register_algorithm
class CQL(OfflineAlgorithm):
    name = "cql"                 # `algo` in configs
    method_id = "cql"            # public id
    family = MODEL_FREE          # or MODEL_BASED
    validation_objective = "validation conservative Q objective"

    def build(self): ...         # networks; the returned module is checkpointed
    def loss(self, batch): ...   # (loss tensor, dict of floats)
    def make_policy(self): ...   # deployed Policy: act() and action_probabilities()
```

Optional hooks cover the rest: `after_update` (target networks, EMA encoders),
`batches` (Dreamer's sequence stream), `update` (a custom optimizer step),
`validate` (extra diagnostics), `state_dict`/`from_checkpoint`, and
`tuning_space` for Optuna. The shared loop in `benchtwm/training.py` does
batching, validation every `offline_log_interval` steps, early stopping,
best-checkpoint selection, gradient clipping (`max_grad_norm`), and logging.

`load_policy(config, checkpoint)` rebuilds the deployed policy of any run.

## Method notes

**DQN.** One-step masked TD targets from a target network updated every
`target_network_frequency` steps with rate `tau`. Deployment is greedy; its
stochastic OPE target is a softmax over Q-values (`policy_temperature`).

**IQL / CQL.** Twin masked Q-networks. IQL fits an expectile value and extracts
the actor by advantage-weighted regression; CQL adds
`logsumexp(Q) - Q(logged action)` over admissible actions. CQL deploys the
minimum of its twin Q-networks.

**CEM world models.** Each family predicts a next state, the immediate reward,
and a continuation probability from `(state, action)`. At decision time the
categorical CEM planner samples action sequences, scores them with the model,
keeps elites, refits per-step action distributions, and executes the first
action of the best sequence. Validation adds open-loop k-step rollout errors
(state, reward, continuation, discounted return) as model-fit diagnostics.
Their OPE action probabilities are marked `diagnostic`, because they come from
the planner's sampled sequences rather than a fixed stochastic policy.

**Dreamer.** RSSM world model with posterior/prior KL, vector reconstruction,
two-hot reward and value heads, continuation, imagined actor learning, slow
value regularization, and replay-value learning, using the source AGC/RMS
optimizer chain. It trains on episode-preserving sequences and carries the
recurrent state across consecutive chunks. Deployment uses the actor's mode;
OPE uses its exact masked categorical distribution. See the
[fidelity audit](dreamer-fidelity.md).
