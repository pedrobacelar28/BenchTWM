import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.algorithms.cem_world_models import CEMWorldModelPolicy, batch_loss, build_model
from src.planners.cem import CategoricalCEMPlanner


def _continuous_batch():
    obs = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.2, 0.2, 0.1, 0.7, 0.3, 0.1],
            [0.9, 0.2, 0.2, 0.2, 0.1, 0.3],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    action_shift = torch.nn.functional.one_hot(actions, num_classes=3).float()
    next_obs = obs + torch.cat([action_shift, action_shift], dim=1) * 0.1
    rewards = next_obs[:, :2].sum(dim=1)
    done = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    return SimpleNamespace(
        obs=obs,
        next_obs=next_obs,
        action=actions,
        reward=rewards,
        done=done,
    )


class TestCEMWorldModels(unittest.TestCase):
    def test_reward_and_continue_heads_use_joint_state_action_inputs(self):
        raw_model = build_model(
            "raw_model",
            obs_dim=6,
            num_actions=3,
            hidden_dim=16,
            action_embed_dim=4,
        )
        self.assertEqual(raw_model.reward_head[0].in_features, 10)
        self.assertEqual(raw_model.continue_head[0].in_features, 10)

        for model_family in ["latent_model", "reconstructive_model"]:
            with self.subTest(model_family=model_family):
                model = build_model(
                    model_family,
                    obs_dim=6,
                    num_actions=3,
                    hidden_dim=16,
                    action_embed_dim=4,
                    latent_dim=8,
                )
                self.assertEqual(model.reward_head[0].in_features, 12)
                self.assertEqual(model.continue_head[0].in_features, 12)

    def test_latent_model_ema_prediction_loss_decreases(self):
        model = build_model(
            "latent_model",
            obs_dim=6,
            num_actions=3,
            hidden_dim=16,
            action_embed_dim=4,
            latent_dim=8,
        )
        batch = _continuous_batch()
        args = SimpleNamespace(
            dynamics_loss_weight=1.0,
            reward_loss_weight=1.0,
            continue_loss_weight=1.0,
            target_encoder_decay=0.99,
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(30):
            loss, _ = batch_loss(model, batch, args, model_family="latent_model")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_encoder(args.target_encoder_decay)
            losses.append(float(loss.item()))
        self.assertLess(losses[-1], losses[0])

    def test_latent_model_requires_target_encoder_decay(self):
        model = build_model(
            "latent_model",
            obs_dim=6,
            num_actions=3,
            hidden_dim=16,
            action_embed_dim=4,
            latent_dim=8,
        )
        batch = _continuous_batch()
        args = SimpleNamespace(
            dynamics_loss_weight=1.0,
            reward_loss_weight=1.0,
            continue_loss_weight=1.0,
        )
        with self.assertRaisesRegex(ValueError, "target_encoder_decay"):
            batch_loss(model, batch, args, model_family="latent_model")

    def test_latent_model_target_encoder_updates(self):
        model = build_model(
            "latent_model",
            obs_dim=6,
            num_actions=3,
            hidden_dim=16,
            action_embed_dim=4,
            latent_dim=8,
        )
        before = [parameter.detach().clone() for parameter in model.target_encoder.parameters()]
        for parameter in model.encoder.parameters():
            parameter.data.add_(1.0)
        model.update_target_encoder(0.5)
        after = list(model.target_encoder.parameters())
        self.assertTrue(any(not torch.allclose(old, new) for old, new in zip(before, after)))

    def test_reconstructive_model_loss_decreases(self):
        model = build_model(
            "reconstructive_model",
            obs_dim=6,
            num_actions=3,
            hidden_dim=16,
            action_embed_dim=4,
            latent_dim=8,
        )
        batch = _continuous_batch()
        args = SimpleNamespace(
            latent_loss_weight=1.0,
            reconstruction_loss_weight=1.0,
            reward_loss_weight=1.0,
            continue_loss_weight=1.0,
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(30):
            loss, _ = batch_loss(model, batch, args, model_family="reconstructive_model")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        self.assertLess(losses[-1], losses[0])

    def test_reward_and_terminal_heads_drive_cem_scores(self):
        for model_family in ["raw_model", "latent_model", "reconstructive_model"]:
            with self.subTest(model_family=model_family):
                kwargs = {
                    "obs_dim": 6,
                    "num_actions": 3,
                    "hidden_dim": 8,
                    "action_embed_dim": 4,
                }
                if model_family != "raw_model":
                    kwargs["latent_dim"] = 5
                model = build_model(model_family, **kwargs)
                for parameter in model.parameters():
                    torch.nn.init.constant_(parameter, 0.0)
                model.reward_head[-1].bias.data.fill_(3.0)
                model.continue_head[-1].bias.data.fill_(8.0)
                planner = CategoricalCEMPlanner(horizon=1, num_actions=3, samples=4, elites=2, iterations=1)
                policy = CEMWorldModelPolicy(model, planner, gamma=1.0)
                rollout_state = model.rollout_state_from_obs(torch.zeros(1, 6)).squeeze(0)
                scores = policy._score_sequences(rollout_state, torch.tensor([[0], [1]], dtype=torch.int64))
                self.assertTrue(torch.allclose(scores, torch.full_like(scores, 3.0), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
