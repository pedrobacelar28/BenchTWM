"""Behavior cloning with a masked categorical actor."""

from __future__ import annotations

from benchtwm.algorithms import register_algorithm
from benchtwm.algorithms.base import MODEL_FREE, OfflineAlgorithm
from benchtwm.algorithms.networks import ActorPolicy, CategoricalActor, hidden_dims


@register_algorithm
class BC(OfflineAlgorithm):
    name = "bc"
    method_id = "bc"
    family = MODEL_FREE
    description = "Behavior cloning: masked cross-entropy on logged actions."
    validation_objective = "validation behavior-cloning cross-entropy"

    def build(self):
        return CategoricalActor(self.spec.obs_dim, self.spec.num_actions, hidden_dims(self.config))

    def loss(self, batch):
        log_prob, entropy = self.model.log_prob_and_entropy(batch.obs, batch.action, batch.action_mask)
        loss = -log_prob.mean()
        return loss, {"bc_loss": float(loss.item()), "entropy": float(entropy.mean().item())}

    def make_policy(self):
        return ActorPolicy(self.model, temperature=float(self.config.get("policy_temperature", 1.0)))
