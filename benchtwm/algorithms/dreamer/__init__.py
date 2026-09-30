"""Offline DreamerV3-style agent (PyTorch port of the pinned JAX reference).

The agent itself lives in :mod:`benchtwm.algorithms.dreamer.agent` and is
covered by the JAX fidelity tests. This module adapts it to the offline
training loop: episode-preserving sequence batches, a recurrent carry across
consecutive chunks, and masked actor deployment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from benchtwm.algorithms import register_algorithm
from benchtwm.algorithms.base import MODEL_BASED, DataSpec, OfflineAlgorithm, cpu_state_dict
from benchtwm.algorithms.dreamer.agent import DreamerAgent, DreamerConfig, DreamerOptimizer
from benchtwm.data.sequences import DreamerSequenceStream
from benchtwm.policies import Policy, greedy

CHECKPOINT_FORMAT = "benchtwm-dreamer-v1"


class DreamerPolicy(Policy):
    ope_probability_contract = {
        "status": "exact",
        "message": "Exact masked categorical probabilities from the recurrent Dreamer actor.",
    }

    def __init__(self, agent: DreamerAgent):
        super().__init__()
        self.agent = agent.eval()
        self.device = next(agent.parameters()).device
        self.obs_dim = int(agent.config.obs_dim)
        self.num_actions = int(agent.config.num_actions)
        self.reset_episode()

    def reset_episode(self) -> None:
        self._state = self.agent.initial(1, device=self.device)
        self._previous_action = torch.zeros(1, dtype=torch.long, device=self.device)
        self._is_first = True

    def action_probabilities(self, obs, info, deterministic: bool = True) -> np.ndarray:
        observation = np.asarray(obs, dtype=np.float32)
        if observation.shape != (self.obs_dim,):
            raise ValueError(f"Dreamer expected an observation of shape {(self.obs_dim,)}, got {observation.shape}")
        mask = np.asarray(info["action_mask"], dtype=np.float32)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.agent.config.autocast_enabled(self.device)):
            token = self.agent.encoder(torch.as_tensor(observation, device=self.device)[None])
            previous = F.one_hot(self._previous_action, self.num_actions).float()
            reset = torch.tensor([self._is_first], dtype=torch.bool, device=self.device)
            self._state, feature = self.agent.rssm.observe_step(self._state, token, previous, reset)
            logits = self.agent.actor_logits(self.agent.feature(feature), torch.as_tensor(mask, device=self.device)[None]).squeeze(0)
        self._is_first = False
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return greedy(probabilities) if deterministic else probabilities

    def observe(self, action: int, reward: float) -> None:
        del reward  # The recurrent carry uses the previous action, not the reward.
        if not 0 <= int(action) < self.num_actions:
            raise ValueError(f"Dreamer received invalid action {action}")
        self._previous_action = torch.tensor([int(action)], dtype=torch.long, device=self.device)


@register_algorithm
class Dreamer(OfflineAlgorithm):
    name = "dreamer"
    method_id = "dreamer"
    family = MODEL_BASED
    description = "Offline DreamerV3-style recurrent world model with imagined actor-critic learning."
    validation_objective = "validation Dreamer world-model, actor, and value objective"

    def build(self):
        self.carry = None
        self.previous_action = None
        return DreamerAgent(DreamerConfig.from_mapping(self.config, obs_dim=self.spec.obs_dim, num_actions=self.spec.num_actions))

    def make_optimizer(self):
        cfg = self.config
        return DreamerOptimizer(
            self.model.optimized_parameters,
            learning_rate=float(cfg.get("learning_rate", 4e-5)),
            agc=float(cfg.get("dreamer_agc", 0.3)),
            eps=float(cfg.get("dreamer_optimizer_eps", 1e-20)),
            beta1=float(cfg.get("dreamer_beta1", 0.9)),
            beta2=float(cfg.get("dreamer_beta2", 0.999)),
            warmup=int(cfg.get("dreamer_warmup", 1000)),
        )

    def _stream(self, dataset, *, seed: int, training: bool) -> DreamerSequenceStream:
        return DreamerSequenceStream(
            dataset,
            batch_size=int(self.config["offline_batch_size"]),
            sequence_length=int(self.config.get("dreamer_batch_length", 64)),
            seed=seed,
            shuffle=training,
            cycle=training,
        )

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.model.config.autocast_enabled(self.device))

    def batches(self, dataset):
        stream = self._stream(dataset, seed=int(self.config["seed"]), training=True)
        while True:
            batch = stream.next_batch(device=self.device)
            if batch is None:
                raise RuntimeError("Cycling Dreamer training stream was unexpectedly exhausted")
            yield batch

    def batch_size(self, batch) -> int:
        return int(batch.valid_records)

    def update(self, batch, step: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        with self._autocast():
            loss, info, carry, previous_action = self.model.loss(batch, self.carry, self.previous_action, update_norms=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Dreamer produced a non-finite loss at step {step}: {float(loss.item())}")
        loss.backward()
        optimizer_info = self.optimizer.step()
        self.model.update_slow_value()
        # The recurrent state carries across consecutive chunks, never across graphs.
        self.carry = {key: value.detach() for key, value in carry.items()}
        self.previous_action = previous_action.detach()
        return {"loss": float(loss.item()), **{key: float(value.item()) for key, value in info.items()}, **optimizer_info}

    @torch.no_grad()
    def validate(self, dataset):
        stream = self._stream(dataset, seed=int(self.config["seed"]) + 200_000, training=False)
        total, records, sums = 0.0, 0, {}
        carry = previous_action = None
        was_training = self.model.training
        self.model.eval()
        devices = [self.device.index] if self.device.type == "cuda" and self.device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.config["seed"]) + 300_000)
            while (batch := stream.next_batch(device=self.device)) is not None:
                with self._autocast():
                    loss, info, carry, previous_action = self.model.loss(batch, carry, previous_action, update_norms=False)
                count = batch.valid_records
                total += float(loss.item()) * count
                records += count
                for key, value in info.items():
                    sums[key] = sums.get(key, 0.0) + float(value.item()) * count
        self.model.train(was_training)
        if records <= 0:
            raise RuntimeError(f"Validation dataset {dataset.path} produced no Dreamer sequences")
        return total / records, {key: value / records for key, value in sums.items()}

    def state_dict(self):
        return {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "dreamer_config": self.model.config.asdict(),
            "model_state_dict": cpu_state_dict(self.model),
        }

    @classmethod
    def from_checkpoint(cls, config, spec: DataSpec, device, checkpoint):
        saved = checkpoint.get("dreamer_config")
        if saved and (int(saved["obs_dim"]) != spec.obs_dim or int(saved["num_actions"]) != spec.num_actions):
            raise ValueError("Dreamer checkpoint dimensions do not match the requested environment")
        # Rebuild with the exact architecture that was trained.
        overrides = {f"dreamer_{key}": value for key, value in (saved or {}).items() if key not in {"obs_dim", "num_actions"}}
        algorithm = cls({**config, **overrides}, spec, device)
        algorithm.load_state_dict(checkpoint)
        return algorithm

    def make_policy(self):
        return DreamerPolicy(self.model)

    @staticmethod
    def tuning_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True),
            "dreamer_imag_length": trial.suggest_int("dreamer_imag_length", 5, 20),
        }
