"""Categorical CEM planner for discrete action sequences."""

from __future__ import annotations

import torch


class CategoricalCEMPlanner:
    def __init__(
        self,
        *,
        horizon: int,
        num_actions: int,
        samples: int,
        elites: int,
        iterations: int,
        alpha: float = 0.2,
        device: str = "cpu",
    ):
        self.horizon = int(horizon)
        self.num_actions = int(num_actions)
        self.samples = int(samples)
        self.elites = int(elites)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.device = device

    def _normalize(self, probs: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(probs, min=1e-8)
        return probs / probs.sum(dim=-1, keepdim=True)

    @staticmethod
    def _restrict(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Exactly zero mass on inadmissible first actions."""

        probs = probs * (mask > 0).float()
        return probs / probs.sum()

    def _initial_probs(self, first_action_mask: torch.Tensor | None) -> torch.Tensor:
        probs = torch.ones(self.horizon, self.num_actions, device=self.device, dtype=torch.float32)
        probs = self._normalize(probs)
        if first_action_mask is not None:
            probs[0] = self._restrict(probs[0], first_action_mask)
        return probs

    def _sample_sequences(self, probs: torch.Tensor) -> torch.Tensor:
        sequences = []
        for t in range(self.horizon):
            dist = torch.distributions.Categorical(probs=probs[t].expand(self.samples, -1))
            sequences.append(dist.sample())
        return torch.stack(sequences, dim=1)

    def plan(self, scorer, first_action_mask: torch.Tensor | None = None) -> tuple[int, dict]:
        probs = self._initial_probs(first_action_mask)
        best_sequence = None
        best_score = None
        for _ in range(self.iterations):
            sequences = self._sample_sequences(probs)
            scores = scorer(sequences)
            elite_idx = torch.topk(scores, k=min(self.elites, sequences.shape[0]), dim=0).indices
            elite_sequences = sequences[elite_idx]
            elite_scores = scores[elite_idx]
            if best_score is None or torch.max(elite_scores) > best_score:
                best_score = torch.max(elite_scores)
                best_sequence = elite_sequences[torch.argmax(elite_scores)]
            counts = []
            for t in range(self.horizon):
                one_hot = torch.nn.functional.one_hot(elite_sequences[:, t], num_classes=self.num_actions).float()
                counts.append(one_hot.mean(dim=0))
            elite_probs = torch.stack(counts, dim=0)
            probs = self._normalize(self.alpha * probs + (1.0 - self.alpha) * elite_probs)
            if first_action_mask is not None:
                probs[0] = self._restrict(self._normalize(probs[0]), first_action_mask)

        if best_sequence is None:
            best_sequence = self._sample_sequences(probs)[0]
            best_score = scorer(best_sequence.unsqueeze(0))[0]
        first_action = int(best_sequence[0].item())
        return first_action, {
            "best_score": float(best_score.item()),
            "horizon": self.horizon,
            "samples": self.samples,
            "elites": self.elites,
            "iterations": self.iterations,
            "first_action_probs": probs[0].detach().cpu().numpy(),
        }
