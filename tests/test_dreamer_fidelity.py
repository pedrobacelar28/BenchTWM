import math
import json
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchtwm.models.dreamer import (
    DreamerAgent,
    DreamerConfig,
    DreamerOptimizer,
    _categorical_kl,
    symlog,
    symexp,
    twohot_bins,
    twohot_loss,
    twohot_prediction,
)
from benchtwm.offline.dreamer_dataset import DreamerSequenceBatch


def _rms(x, scale, eps=1e-4):
    return x * (1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)) * scale


def _linear(module, x):
    result = x @ module.weight.detach().numpy().T
    if module.bias is not None:
        result += module.bias.detach().numpy()
    return result


def _silu(x):
    return x / (1.0 + np.exp(-x))


class TestDreamerFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = Path(__file__).parent / "fixtures" / "dreamer_jax_reference.json"
        cls.jax_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def test_pinned_jax_fixture_matches_pytorch_modules(self):
        self.assertEqual(self.jax_fixture["jax_version"], "0.4.33")
        values = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        torch.testing.assert_close(symlog(values), torch.tensor(self.jax_fixture["symlog"]), atol=1e-6, rtol=1e-6)

        cfg = DreamerConfig(
            obs_dim=3,
            num_actions=5,
            deter=8,
            hidden=3,
            stoch=2,
            classes=4,
            blocks=4,
            units=8,
            enc_layers=1,
            dec_layers=1,
            rssm_img_layers=1,
            bins=5,
            imag_length=2,
        )
        rssm = DreamerAgent(cfg).rssm

        def set_linear(module, index):
            source = torch.arange(module.weight.numel(), dtype=torch.float32).reshape(
                module.weight.shape[1], module.weight.shape[0]
            )
            source = (source + index + 1) / (100.0 + index)
            module.weight.data.copy_(source.T)
            module.bias.data.copy_(torch.arange(module.bias.numel(), dtype=torch.float32) / (50.0 + index))

        for index, ((linear, norm), scale) in enumerate(
            zip(
                rssm.dyn_inputs,
                ([1.0, 0.8, 1.2], [0.9, 1.1, 1.0], [1.2, 1.0, 0.7]),
            )
        ):
            set_linear(linear, index)
            norm.scale.data.copy_(torch.tensor(scale))

        linear, norm = rssm.dyn_layers[0]
        shape = linear.weight.shape
        weight = (torch.arange(linear.weight.numel(), dtype=torch.float32).reshape(shape) + 4) / 203.0
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(torch.arange(8, dtype=torch.float32) / 83.0)
        norm.scale.data.copy_(torch.linspace(0.8, 1.2, 8))
        linear = rssm.gru
        weight = (torch.arange(linear.weight.numel(), dtype=torch.float32).reshape(linear.weight.shape) + 5) / 204.0
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(torch.arange(24, dtype=torch.float32) / 84.0)

        deter = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10.0 - 0.5
        stoch = F.one_hot(torch.tensor([[0, 2], [1, 3]]), 4).float()
        action = F.one_hot(torch.tensor([1, 3]), 5).float()
        actual = rssm._core(deter, stoch, action)
        expected = torch.tensor(self.jax_fixture["rssm_core"])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_symlog_symexp_reference_formula(self):
        values = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        expected = np.sign(values.numpy()) * np.log1p(np.abs(values.numpy()))
        np.testing.assert_allclose(symlog(values).numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(symexp(symlog(values)).numpy(), values.numpy(), rtol=1e-5, atol=1e-5)

    def test_twohot_matches_reference_interpolation_and_symmetric_prediction(self):
        bins = twohot_bins(5)
        logits = torch.tensor([[0.2, -0.3, 0.1, 0.4, -0.1]], dtype=torch.float32)
        target = torch.tensor([0.5], dtype=torch.float32)
        below = int(torch.nonzero(bins <= target.item())[-1])
        above = int(torch.nonzero(bins >= target.item())[0])
        db = abs(float(bins[below]) - target.item())
        da = abs(float(bins[above]) - target.item())
        weights = np.zeros(5, dtype=np.float32)
        weights[below] += da / (db + da)
        weights[above] += db / (db + da)
        logprob = F.log_softmax(logits, dim=-1).numpy()[0]
        expected_loss = -float(np.sum(weights * logprob))
        self.assertAlmostEqual(float(twohot_loss(logits, target, bins)), expected_loss, places=6)
        uniform = torch.zeros(3, 5)
        torch.testing.assert_close(twohot_prediction(uniform, bins), torch.zeros(3), atol=1e-7, rtol=0)

    def test_unimix_categorical_kl_matches_numpy(self):
        lhs = torch.tensor([[[1.0, -0.5, 0.2], [0.1, 0.4, -0.7]]])
        rhs = torch.tensor([[[0.0, 0.2, -0.1], [0.8, -0.2, 0.0]]])
        unimix = 0.01

        def probs(x):
            exp = np.exp(x - x.max(axis=-1, keepdims=True))
            value = exp / exp.sum(axis=-1, keepdims=True)
            return (1 - unimix) * value + unimix / value.shape[-1]

        p, q = probs(lhs.numpy()), probs(rhs.numpy())
        expected = np.sum(p * (np.log(p) - np.log(q)), axis=(-1, -2))
        np.testing.assert_allclose(_categorical_kl(lhs, rhs, unimix).numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_blocked_rssm_core_matches_independent_numpy_translation(self):
        torch.manual_seed(7)
        cfg = DreamerConfig(
            obs_dim=3,
            num_actions=5,
            deter=8,
            hidden=3,
            stoch=2,
            classes=4,
            blocks=4,
            units=8,
            enc_layers=1,
            dec_layers=1,
            rssm_img_layers=1,
            bins=5,
            imag_length=2,
        )
        rssm = DreamerAgent(cfg).rssm
        deter = torch.randn(2, 8)
        stoch = F.one_hot(torch.randint(0, 4, (2, 2)), 4).float()
        action = F.one_hot(torch.tensor([1, 3]), 5).float()
        actual = rssm._core(deter, stoch, action).detach().numpy()

        inputs = []
        for value, (linear, norm) in zip((deter.numpy(), stoch.numpy().reshape(2, -1), action.numpy()), rssm.dyn_inputs):
            inputs.append(_silu(_rms(_linear(linear, value), norm.scale.detach().numpy())))
        grouped = deter.numpy().reshape(2, cfg.blocks, cfg.deter // cfg.blocks)
        repeated = np.repeat(np.concatenate(inputs, axis=-1)[:, None], cfg.blocks, axis=1)
        x = np.concatenate([grouped, repeated], axis=-1).reshape(2, -1)
        for linear, norm in rssm.dyn_layers:
            weight = linear.weight.detach().numpy()
            block = x.reshape(2, linear.blocks, linear.in_block)
            x = np.einsum("bki,kio->bko", block, weight).reshape(2, -1) + linear.bias.detach().numpy()
            x = _silu(_rms(x, norm.scale.detach().numpy()))
        linear = rssm.gru
        gates = np.einsum(
            "bki,kio->bko", x.reshape(2, linear.blocks, linear.in_block), linear.weight.detach().numpy()
        ).reshape(2, -1) + linear.bias.detach().numpy()
        gates = gates.reshape(2, cfg.blocks, 3 * (cfg.deter // cfg.blocks))
        reset, candidate, update = np.split(gates, 3, axis=-1)
        reset = 1.0 / (1.0 + np.exp(-reset.reshape(2, -1)))
        candidate = np.tanh(reset * candidate.reshape(2, -1))
        update = 1.0 / (1.0 + np.exp(-(update.reshape(2, -1) - 1.0)))
        expected = update * candidate + (1.0 - update) * deter.numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_lambda_return_matches_jax_loop(self):
        cfg = DreamerConfig(obs_dim=2, num_actions=3, deter=8, hidden=4, stoch=2, classes=4, blocks=4, units=8, bins=5)
        agent = DreamerAgent(cfg)
        last = torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)
        term = torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)
        rew = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        val = torch.tensor([[0.5, 0.6, 0.7, 0.8]])
        boot = torch.tensor([[0.5, 0.6, 0.7, 0.8]])
        actual = agent._lambda_return(last, term, rew, val, boot, 0.99, 0.95).numpy()
        live = (1 - term.numpy().astype(np.float32))[:, 1:] * 0.99
        cont = (1 - last.numpy().astype(np.float32))[:, 1:] * 0.95
        interm = rew.numpy()[:, 1:] + (1 - cont) * live * boot.numpy()[:, 1:]
        values = [boot.numpy()[:, -1]]
        for time in reversed(range(live.shape[1])):
            values.append(interm[:, time] + live[:, time] * cont[:, time] * values[-1])
        expected = np.stack(list(reversed(values))[:-1], axis=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_optimizer_chain_matches_first_optax_style_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = torch.tensor([3.0, 4.0])
        optimizer = DreamerOptimizer([parameter], learning_rate=0.01, agc=0.3, beta1=0.9, beta2=0.999, eps=1e-20, warmup=0)
        original = parameter.detach().numpy().copy()
        gradient = np.array([3.0, 4.0], dtype=np.float32)
        upper = 0.3 * np.linalg.norm(original)
        gradient *= min(1.0, upper / np.linalg.norm(gradient))
        # First-step RMS and momentum bias corrections both reduce to identity.
        expected = original - 0.01 * gradient / np.sqrt(np.square(gradient))
        optimizer.step()
        np.testing.assert_allclose(parameter.detach().numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_complete_tiny_agent_loss_and_gradients_are_finite(self):
        torch.manual_seed(11)
        cfg = DreamerConfig(
            obs_dim=3,
            num_actions=5,
            deter=16,
            hidden=8,
            stoch=2,
            classes=4,
            blocks=4,
            units=8,
            enc_layers=1,
            dec_layers=1,
            rssm_img_layers=1,
            bins=5,
            imag_length=3,
        )
        agent = DreamerAgent(cfg)
        batch = DreamerSequenceBatch(
            observations=torch.randn(2, 5, 3),
            actions=torch.randint(0, 5, (2, 5)),
            rewards=torch.randn(2, 5),
            is_first=torch.tensor([[1, 0, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool),
            is_last=torch.tensor([[0, 0, 1, 0, 1], [0, 0, 0, 0, 1]], dtype=torch.bool),
            is_terminal=torch.tensor([[0, 0, 1, 0, 1], [0, 0, 0, 0, 1]], dtype=torch.bool),
            valid_mask=torch.ones(2, 5),
            action_masks=torch.ones(2, 5, 5),
        )
        loss, metrics, _, _ = agent.loss(batch, update_norms=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            {key for key in metrics if key.startswith("loss_")},
            {"loss_dyn", "loss_rep", "loss_rec", "loss_rew", "loss_con", "loss_policy", "loss_value", "loss_repval"},
        )
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in agent.optimized_parameters if parameter.grad is not None))


if __name__ == "__main__":
    unittest.main()
