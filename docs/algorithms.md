# Algorithms

## Scope

The suite implements six local agent families. They share observation backends, action masks, checkpoint loading, direct evaluation, and `action_probabilities(obs, info, deterministic=...)` for OPE.

This differs from the upstream EpiCare paper. That paper studies AWAC, EDAC, TD3+BC, IQL, CQL, behavior cloning, and DQN. This suite vendors EpiCare's environment and policies but implements the methods below.

| Family | `algo` | Online mode | Offline mode |
|---|---|---|---|
| Deep Q-learning | `dqn` | Replay-buffer DQN with target network | Fitted Q updates on fixed transitions |
| Proximal policy optimization | `ppo` | On-policy rollout PPO | Logged PPO-style objective plus behavior cloning |
| Raw world model | `raw_model` | Learned dynamics with CEM MPC | Fixed-data learned model with CEM MPC |
| Latent world model | `latent_model` | EMA-target latent dynamics with CEM MPC | Same objective on fixed data |
| Reconstructive world model | `reconstructive_model` | Latent dynamics plus reconstruction with CEM MPC | Same objective on fixed data |
| DreamerV3-style | `dreamer` | Not available | Recurrent world model and imagined actor-critic learning on fixed trajectories |

## Observation Encoders and Masks

DQN and PPO select `policy_architecture`. `tabular` is a bias-free linear table over ICU one-hot state IDs; `mlp` is a feed-forward network for continuous vectors and is used by EpiCare. All policy logits and Q-values are masked with `action_mask`. This prevents ICU policies from selecting inadmissible actions and leaves EpiCare's all-ones mask unchanged.

World models receive a selected categorical action as an embedding and use the chosen observation backend as model input.

## DQN

Online DQN collects masked epsilon-greedy transitions in a replay buffer. It optimizes squared TD error against a target network and applies soft target updates controlled by `tau` and `target_network_frequency`.

Offline DQN samples only `train.npz`. Its objective is the same masked one-step TD target, while validation loss is measured on `val.npz`. The best checkpoint is selected by validation loss. The deployed policy is greedy by default; its non-deterministic OPE probabilities are a Boltzmann distribution over masked Q-values.

## PPO

Online PPO stores a rollout, estimates generalized advantages, and optimizes a clipped policy objective with value and entropy terms.

Offline PPO is not conventional on-policy PPO. It evaluates logged actions under the current policy, forms a clipped likelihood ratio using `behavior_logprobs`, and adds a behavior-cloning penalty. Its metrics include total, policy, value, BC, and entropy losses. Describe this method as `ppo_logged_offline`, not as an on-policy offline rollout method.

## CEM World Models

Every world-model family predicts a next-state representation, immediate reward, and continuation probability conditioned on state and action. At decision time, `CategoricalCEMPlanner` samples categorical action sequences, scores them under the learned model, keeps elites, updates action distributions, and deploys the first action from the best sequence. The first-step action mask is respected.

| Family | State-prediction target | Main losses |
|---|---|---|
| Raw | Next observation directly | Dynamics, reward, continuation |
| Latent | EMA target-encoder latent | Latent dynamics, reward, continuation |
| Reconstructive | Latent target plus decoded next observation | Latent, reconstruction, reward, continuation |

Offline world-model validation computes multi-step rollout errors at horizons derived from `cem_horizon`: state error, reward error, continuation loss, discounted-return error, and cumulative state error. These are model-fit diagnostics, not direct policy-quality estimates.

## Offline DreamerV3-style

`dreamer` is a PyTorch port of the pinned JAX DreamerV3 vector/discrete-action
path. It reconstructs state records from the transition schema, keeps RSSM
carry across consecutive 64-step chunks, and resets only at real episode
boundaries. The loss includes posterior/prior KL, vector reconstruction,
two-hot reward and value heads, continuation prediction, imagined categorical
actor learning, slow-value regularization, lambda returns, and replay-value
learning. The optimizer preserves the JAX AGC, RMS scaling, momentum, and
warmup chain.

Only `state_vector` is encoded. The categorical stochastic RSSM state and
one-hot action embedding are internal model variables, not one-hot agent
observations. At deployment, the environment mask is applied to actor logits;
the unmasked imagination equations remain identical to the source algorithm.
Direct evaluation uses the actor mode, while stochastic OPE uses its exact
masked categorical distribution. See [Dreamer fidelity audit](dreamer-fidelity.md).

## Training Metrics

`offline_training_history.csv` is dataset-only and records optimization and validation losses by gradient step. `offline_eval_history.csv` periodically runs the current checkpoint in the simulator and records common and environment metrics. Online training records return, success, duration, action stability, and environment metric summaries. `survival_rate` remains a compatibility alias for `success_rate`; new analysis should use `success_rate`.

## Adding a Method

```python
def train(config: dict, output_dir):
    return {
        "metrics": metrics,
        "checkpoint_last": checkpoint_last,
        "checkpoint_best": checkpoint_best,
    }

def load_policy(config: dict, checkpoint: dict):
    return policy  # policy.act(...) and policy.action_probabilities(...)
```

Register online training/loading in `src/algorithms/registry.py`. Register an offline trainer in `src/offline/trainers.py` when fixed-data training is supported. Then add dev configs, ICU/EpiCare smoke tests, and an OPE test for the policy-probability contract. The full checklist is in [extending.md](extending.md).
