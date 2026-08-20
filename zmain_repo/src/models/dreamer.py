"""PyTorch port of the vector/discrete-action DreamerV3 agent.

The equations and module ordering follow the JAX reference pinned in
``THIRD_PARTY_NOTICES.md``.  BenchTWM-specific data loading and action masking
live outside the mathematical core so they do not change the unmasked Dreamer
losses.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def _trunc_normal_(tensor: torch.Tensor, *, fan_in: int, scale: float = 1.0) -> None:
    # JAX samples N(0, 1) truncated to [-2, 2] and multiplies by 1.1368.
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
        tensor.mul_(1.1368 * scale / math.sqrt(max(1, fan_in)))


class DreamerLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, outscale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        _trunc_normal_(self.weight, fan_in=in_features, scale=outscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class BlockLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, blocks: int):
        super().__init__()
        if in_features % blocks or out_features % blocks:
            raise ValueError("BlockLinear dimensions must be divisible by blocks")
        self.blocks = int(blocks)
        self.in_block = in_features // blocks
        self.out_block = out_features // blocks
        self.weight = nn.Parameter(torch.empty(blocks, self.in_block, self.out_block))
        self.bias = nn.Parameter(torch.zeros(out_features))
        _trunc_normal_(self.weight, fan_in=in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        grouped = x.reshape(*shape, self.blocks, self.in_block)
        output = torch.einsum("...ki,kio->...ko", grouped, self.weight)
        return output.reshape(*shape, self.blocks * self.out_block) + self.bias


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps) * self.scale.float()
        return value.to(dtype)


class DenseStack(nn.Module):
    def __init__(self, in_features: int, layers: int, units: int):
        super().__init__()
        modules = []
        current = in_features
        for _ in range(int(layers)):
            modules.append(nn.ModuleList([DreamerLinear(current, units), RMSNorm(units)]))
            current = units
        self.layers = nn.ModuleList(modules)
        self.out_features = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, norm in self.layers:
            x = F.silu(norm(linear(x)))
        return x


def _unimix_logits(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    if not unimix:
        return logits.float()
    probs = torch.softmax(logits.float(), dim=-1)
    probs = (1.0 - unimix) * probs + unimix / probs.shape[-1]
    return torch.log(probs)


def _sample_onehot(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    mixed = _unimix_logits(logits, unimix)
    probs = torch.softmax(mixed, dim=-1)
    index = torch.distributions.Categorical(probs=probs).sample()
    hard = F.one_hot(index, mixed.shape[-1]).to(torch.float32)
    return hard.detach() + probs - probs.detach()


def _categorical_kl(lhs: torch.Tensor, rhs: torch.Tensor, unimix: float) -> torch.Tensor:
    lhs = _unimix_logits(lhs, unimix)
    rhs = _unimix_logits(rhs, unimix)
    lhs_logprob = F.log_softmax(lhs, dim=-1)
    rhs_logprob = F.log_softmax(rhs, dim=-1)
    return (lhs_logprob.exp() * (lhs_logprob - rhs_logprob)).sum(dim=-1).sum(dim=-1)


def _categorical_entropy(logits: torch.Tensor, unimix: float = 0.0) -> torch.Tensor:
    logits = _unimix_logits(logits, unimix)
    logprob = F.log_softmax(logits, dim=-1)
    return -(logprob.exp() * logprob).sum(dim=-1)


def twohot_bins(count: int, *, device: torch.device | None = None) -> torch.Tensor:
    if count % 2:
        half = symexp(torch.linspace(-20.0, 0.0, (count - 1) // 2 + 1, dtype=torch.float32, device=device))
        return torch.cat([half, -torch.flip(half[:-1], dims=[0])])
    half = symexp(torch.linspace(-20.0, 0.0, count // 2, dtype=torch.float32, device=device))
    return torch.cat([half, -torch.flip(half, dims=[0])])


def twohot_prediction(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    n = logits.shape[-1]
    if n % 2:
        middle = (n - 1) // 2
        center = (probs[..., middle : middle + 1] * bins[middle : middle + 1]).sum(dim=-1)
        paired = (
            torch.flip(probs[..., :middle] * bins[:middle], dims=[-1])
            + probs[..., middle + 1 :] * bins[middle + 1 :]
        ).sum(dim=-1)
        return center + paired
    middle = n // 2
    return (
        torch.flip(probs[..., :middle] * bins[:middle], dims=[-1])
        + probs[..., middle:] * bins[middle:]
    ).sum(dim=-1)


def twohot_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    target = target.detach().float()
    flat = target.reshape(-1)
    above = torch.searchsorted(bins, flat, right=False).clamp(0, len(bins) - 1)
    below = (above - 1).clamp(0, len(bins) - 1)
    # searchsorted returns the exact bin as ``above``; JAX uses the same bin for
    # both sides in this case.
    exact = bins[above] == flat
    below = torch.where(exact, above, below)
    equal = below == above
    dist_below = torch.where(equal, torch.ones_like(flat), torch.abs(bins[below] - flat))
    dist_above = torch.where(equal, torch.ones_like(flat), torch.abs(bins[above] - flat))
    total = dist_below + dist_above
    weight_below = dist_above / total
    weight_above = dist_below / total
    target_probs = F.one_hot(below, len(bins)).float() * weight_below[:, None]
    target_probs += F.one_hot(above, len(bins)).float() * weight_above[:, None]
    target_probs = target_probs.reshape(*target.shape, len(bins))
    return -(target_probs * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)


class RunningNormalize(nn.Module):
    def __init__(
        self,
        impl: str,
        *,
        rate: float = 0.01,
        limit: float = 1e-8,
        perclo: float = 5.0,
        perchi: float = 95.0,
        debias: bool = True,
    ):
        super().__init__()
        if impl not in {"none", "meanstd", "perc"}:
            raise ValueError(f"Unsupported normalization {impl!r}")
        self.impl = impl
        self.rate = float(rate)
        self.limit = float(limit)
        self.perclo = float(perclo)
        self.perchi = float(perchi)
        self.debias = bool(debias)
        self.register_buffer("corr", torch.zeros(()))
        self.register_buffer("first", torch.zeros(()))
        self.register_buffer("second", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        value = x.detach().float()
        if self.impl == "meanstd":
            first, second = value.mean(), value.square().mean()
        elif self.impl == "perc":
            first = torch.quantile(value.flatten(), self.perclo / 100.0)
            second = torch.quantile(value.flatten(), self.perchi / 100.0)
        else:
            return
        self.first.mul_(1.0 - self.rate).add_(first, alpha=self.rate)
        self.second.mul_(1.0 - self.rate).add_(second, alpha=self.rate)
        if self.debias:
            self.corr.mul_(1.0 - self.rate).add_(self.rate)

    def stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.impl == "none":
            return self.first.new_zeros(()), self.first.new_ones(())
        correction = self.first.new_ones(())
        if self.debias:
            correction = 1.0 / torch.maximum(self.first.new_tensor(self.rate), self.corr)
        if self.impl == "meanstd":
            mean = self.first * correction
            std = torch.sqrt(torch.relu(self.second * correction - mean.square()))
            return mean, torch.maximum(self.first.new_tensor(self.limit), std)
        low, high = self.first * correction, self.second * correction
        return low.detach(), torch.maximum(self.first.new_tensor(self.limit), high - low).detach()

    def forward(self, x: torch.Tensor, *, update: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if update:
            self.update(x)
        return self.stats()


@dataclass(frozen=True)
class DreamerConfig:
    obs_dim: int
    num_actions: int
    compute_dtype: str = "float32"
    deter: int = 8192
    hidden: int = 1024
    stoch: int = 32
    classes: int = 64
    blocks: int = 8
    enc_layers: int = 3
    dec_layers: int = 3
    units: int = 1024
    rssm_img_layers: int = 2
    rssm_obs_layers: int = 1
    rssm_dyn_layers: int = 1
    unimix: float = 0.01
    free_nats: float = 1.0
    bins: int = 255
    imag_length: int = 15
    imag_last: int = 0
    horizon: int = 333
    contdisc: bool = True
    lam: float = 0.95
    actent: float = 3e-4
    slowreg: float = 1.0
    slowtar: bool = False
    slowvalue_rate: float = 0.02
    reward_grad: bool = True
    repval_loss: bool = True
    repval_grad: bool = True
    ac_grads: bool = False
    rec_scale: float = 1.0
    rew_scale: float = 1.0
    con_scale: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    policy_scale: float = 1.0
    value_scale: float = 1.0
    repval_scale: float = 0.3

    @classmethod
    def from_mapping(cls, config: dict[str, Any], *, obs_dim: int, num_actions: int) -> "DreamerConfig":
        values: dict[str, Any] = {"obs_dim": int(obs_dim), "num_actions": int(num_actions)}
        for field in cls.__dataclass_fields__:
            if field in values:
                continue
            key = f"dreamer_{field}"
            if key in config:
                values[field] = config[key]
        return cls(**values)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    def autocast_enabled(self, device: torch.device) -> bool:
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported Dreamer compute dtype {self.compute_dtype!r}")
        return self.compute_dtype == "bfloat16" and device.type == "cuda"


class VectorEncoder(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.mlp = DenseStack(config.obs_dim, config.enc_layers, config.units)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(symlog(observation.float()))


class VectorDecoder(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        feature_dim = config.deter + config.stoch * config.classes
        self.mlp = DenseStack(feature_dim, config.dec_layers, config.units)
        self.output = DreamerLinear(self.mlp.out_features, config.obs_dim, outscale=1.0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.output(self.mlp(feature))


class MLPHead(nn.Module):
    def __init__(self, in_features: int, *, layers: int, units: int, out_features: int, outscale: float):
        super().__init__()
        self.mlp = DenseStack(in_features, layers, units)
        self.output = DreamerLinear(self.mlp.out_features, out_features, outscale=outscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.mlp(x))


class RSSM(nn.Module):
    def __init__(self, config: DreamerConfig):
        super().__init__()
        if config.deter % config.blocks:
            raise ValueError("dreamer_deter must be divisible by dreamer_blocks")
        self.config = config
        h, d, s, c, a = config.hidden, config.deter, config.stoch, config.classes, config.num_actions
        self.dyn_inputs = nn.ModuleList([
            nn.ModuleList([DreamerLinear(d, h), RMSNorm(h)]),
            nn.ModuleList([DreamerLinear(s * c, h), RMSNorm(h)]),
            nn.ModuleList([DreamerLinear(a, h), RMSNorm(h)]),
        ])
        core_in = d + config.blocks * (3 * h)
        self.dyn_layers = nn.ModuleList()
        for _ in range(config.rssm_dyn_layers):
            self.dyn_layers.append(nn.ModuleList([BlockLinear(core_in, d, config.blocks), RMSNorm(d)]))
            core_in = d
        self.gru = BlockLinear(core_in, 3 * d, config.blocks)
        self.obs_layers = nn.ModuleList()
        obs_in = d + config.units
        for _ in range(config.rssm_obs_layers):
            self.obs_layers.append(nn.ModuleList([DreamerLinear(obs_in, h), RMSNorm(h)]))
            obs_in = h
        self.obs_logits = DreamerLinear(obs_in, s * c, outscale=1.0)
        self.prior_layers = nn.ModuleList()
        prior_in = d
        for _ in range(config.rssm_img_layers):
            self.prior_layers.append(nn.ModuleList([DreamerLinear(prior_in, h), RMSNorm(h)]))
            prior_in = h
        self.prior_logits = DreamerLinear(prior_in, s * c, outscale=1.0)

    def initial(self, batch_size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
        cfg = self.config
        dtype = torch.bfloat16 if cfg.autocast_enabled(device) else torch.float32
        return {
            "deter": torch.zeros(batch_size, cfg.deter, device=device, dtype=dtype),
            "stoch": torch.zeros(batch_size, cfg.stoch, cfg.classes, device=device, dtype=dtype),
        }

    def _core(self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        stoch = stoch.to(deter.dtype)
        action = action.to(deter.dtype)
        stoch = stoch.reshape(stoch.shape[0], -1)
        action = action / torch.maximum(torch.ones_like(action), action.abs()).detach()
        parts = []
        for value, (linear, norm) in zip((deter, stoch, action), self.dyn_inputs):
            parts.append(F.silu(norm(linear(value))))
        grouped_deter = deter.reshape(deter.shape[0], cfg.blocks, cfg.deter // cfg.blocks)
        repeated = torch.cat(parts, dim=-1)[:, None, :].expand(-1, cfg.blocks, -1)
        x = torch.cat([grouped_deter, repeated], dim=-1).reshape(deter.shape[0], -1)
        for linear, norm in self.dyn_layers:
            x = F.silu(norm(linear(x)))
        gates = self.gru(x).reshape(deter.shape[0], cfg.blocks, 3 * (cfg.deter // cfg.blocks))
        reset, candidate, update = torch.chunk(gates, 3, dim=-1)
        reset = reset.reshape(deter.shape[0], -1).sigmoid()
        candidate = torch.tanh(reset * candidate.reshape(deter.shape[0], -1))
        update = torch.sigmoid(update.reshape(deter.shape[0], -1) - 1.0)
        return update * candidate + (1.0 - update) * deter

    def prior(self, deter: torch.Tensor) -> torch.Tensor:
        x = deter
        for linear, norm in self.prior_layers:
            x = F.silu(norm(linear(x)))
        cfg = self.config
        return self.prior_logits(x).reshape(*x.shape[:-1], cfg.stoch, cfg.classes)

    def observe_step(
        self,
        state: dict[str, torch.Tensor],
        token: torch.Tensor,
        previous_action: torch.Tensor,
        reset: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        dtype = torch.bfloat16 if self.config.autocast_enabled(state["deter"].device) else torch.float32
        token = token.to(dtype)
        previous_action = previous_action.to(dtype)
        state = {key: value.to(dtype) for key, value in state.items()}
        keep = (~reset).to(dtype)
        deter = state["deter"] * keep[:, None]
        stoch = state["stoch"] * keep[:, None, None]
        action = previous_action * keep[:, None]
        deter = self._core(deter, stoch, action)
        x = torch.cat([deter, token.reshape(token.shape[0], -1)], dim=-1)
        for linear, norm in self.obs_layers:
            x = F.silu(norm(linear(x)))
        cfg = self.config
        logits = self.obs_logits(x).reshape(x.shape[0], cfg.stoch, cfg.classes)
        stoch = _sample_onehot(logits, cfg.unimix).to(deter.dtype)
        state = {"deter": deter, "stoch": stoch}
        feature = {"deter": deter, "stoch": stoch, "logit": logits}
        return state, feature

    def observe(
        self,
        initial: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        previous_actions: torch.Tensor,
        resets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        state = initial
        entries: dict[str, list[torch.Tensor]] = {"deter": [], "stoch": [], "logit": []}
        for time in range(tokens.shape[1]):
            state, feature = self.observe_step(state, tokens[:, time], previous_actions[:, time], resets[:, time])
            for key in entries:
                entries[key].append(feature[key])
        return state, {key: torch.stack(values, dim=1) for key, values in entries.items()}

    def imagine_step(
        self, state: dict[str, torch.Tensor], action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        deter = self._core(state["deter"], state["stoch"], action)
        logits = self.prior(deter)
        stoch = _sample_onehot(logits, self.config.unimix).to(deter.dtype)
        state = {"deter": deter, "stoch": stoch}
        return state, {"deter": deter, "stoch": stoch, "logit": logits}


class DreamerAgent(nn.Module):
    """Vector observation, categorical action specialization of DreamerV3."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        feature_dim = config.deter + config.stoch * config.classes
        self.encoder = VectorEncoder(config)
        self.rssm = RSSM(config)
        self.decoder = VectorDecoder(config)
        self.reward_head = MLPHead(feature_dim, layers=1, units=config.units, out_features=config.bins, outscale=0.0)
        self.continue_head = MLPHead(feature_dim, layers=1, units=config.units, out_features=1, outscale=1.0)
        self.policy_head = MLPHead(feature_dim, layers=3, units=config.units, out_features=config.num_actions, outscale=0.01)
        self.value_head = MLPHead(feature_dim, layers=3, units=config.units, out_features=config.bins, outscale=0.0)
        self.slow_value_head = copy.deepcopy(self.value_head)
        for parameter in self.slow_value_head.parameters():
            parameter.requires_grad_(False)
        self.return_norm = RunningNormalize("perc", rate=0.01, limit=1.0, perclo=5.0, perchi=95.0, debias=False)
        self.value_norm = RunningNormalize("none", rate=0.01, limit=1e-8)
        self.advantage_norm = RunningNormalize("none", rate=0.01, limit=1e-8)
        self.register_buffer("bins", twohot_bins(config.bins))

    @property
    def optimized_parameters(self):
        modules = [
            self.encoder,
            self.rssm,
            self.decoder,
            self.reward_head,
            self.continue_head,
            self.policy_head,
            self.value_head,
        ]
        for module in modules:
            yield from module.parameters()

    def initial(self, batch_size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
        return self.rssm.initial(batch_size, device=device)

    def feature(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        stoch = state["stoch"].reshape(*state["stoch"].shape[:-2], -1)
        return torch.cat([state["deter"], stoch], dim=-1)

    def decoder_feature(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        # rssm.Decoder concatenates ('stoch', 'deter'), while Agent heads use
        # feat2tensor's ('deter', 'stoch') order.
        stoch = state["stoch"].reshape(*state["stoch"].shape[:-2], -1)
        return torch.cat([stoch, state["deter"]], dim=-1)

    def actor_logits(self, feature: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.policy_head(feature).float()
        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool, device=logits.device)
            if not torch.all(mask.any(dim=-1)):
                raise ValueError("Dreamer received an action mask with no valid actions")
            logits = logits.masked_fill(~mask, -1e30)
        return logits

    @torch.no_grad()
    def update_slow_value(self) -> None:
        rate = self.config.slowvalue_rate
        for source, target in zip(self.value_head.parameters(), self.slow_value_head.parameters()):
            target.mul_(1.0 - rate).add_(source, alpha=rate)

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(value.dtype).expand_as(value)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def _lambda_return(
        self,
        last: torch.Tensor,
        terminal: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        lam: float,
    ) -> torch.Tensor:
        returns = [bootstrap[:, -1]]
        live = (1.0 - terminal.float())[:, 1:] * discount
        cont = (1.0 - last.float())[:, 1:] * lam
        intermediate = reward[:, 1:] + (1.0 - cont) * live * bootstrap[:, 1:]
        for time in reversed(range(live.shape[1])):
            returns.append(intermediate[:, time] + live[:, time] * cont[:, time] * returns[-1])
        return torch.stack(list(reversed(returns))[:-1], dim=1)

    def _imagine(
        self, starts: dict[str, torch.Tensor], length: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        state = starts
        features: dict[str, list[torch.Tensor]] = {"deter": [], "stoch": [], "logit": []}
        actions = []
        for _ in range(length):
            detached = {key: value.detach() for key, value in state.items()}
            logits = self.actor_logits(self.feature(detached))
            action_index = torch.distributions.Categorical(logits=logits).sample()
            action = F.one_hot(action_index, self.config.num_actions).float()
            state, feature = self.rssm.imagine_step(state, action)
            actions.append(action_index)
            for key in features:
                features[key].append(feature[key])
        return (
            {key: torch.stack(values, dim=1) for key, values in features.items()},
            torch.stack(actions, dim=1),
        )

    def loss(
        self,
        batch,
        carry: dict[str, torch.Tensor] | None = None,
        previous_action: torch.Tensor | None = None,
        *,
        update_norms: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        cfg = self.config
        obs = batch.observations.float()
        B, T, _ = obs.shape
        device = obs.device
        if carry is None:
            carry = self.initial(B, device=device)
        if previous_action is None:
            previous_action = torch.zeros(B, dtype=torch.long, device=device)
        actions_onehot = F.one_hot(batch.actions, cfg.num_actions).float()
        previous_onehot = F.one_hot(previous_action, cfg.num_actions).float()
        previous_actions = torch.cat([previous_onehot[:, None], actions_onehot[:, :-1]], dim=1)

        tokens = self.encoder(obs)
        final_state, posterior = self.rssm.observe(carry, tokens, previous_actions, batch.is_first)
        prior_logits = self.rssm.prior(posterior["deter"])
        dyn_loss = _categorical_kl(posterior["logit"].detach(), prior_logits, cfg.unimix)
        rep_loss = _categorical_kl(posterior["logit"], prior_logits.detach(), cfg.unimix)
        if cfg.free_nats:
            dyn_loss = torch.maximum(dyn_loss, dyn_loss.new_tensor(cfg.free_nats))
            rep_loss = torch.maximum(rep_loss, rep_loss.new_tensor(cfg.free_nats))

        feature = self.feature(posterior)
        reconstruction = self.decoder(self.decoder_feature(posterior))
        reconstruction_loss = (reconstruction - symlog(obs).detach()).square().sum(dim=-1)
        reward_input = feature if cfg.reward_grad else feature.detach()
        reward_logits = self.reward_head(reward_input)
        reward_loss = twohot_loss(reward_logits, batch.rewards, self.bins)
        continue_target = (~batch.is_terminal).float()
        if cfg.contdisc:
            continue_target = continue_target * (1.0 - 1.0 / cfg.horizon)
        continue_logits = self.continue_head(feature).squeeze(-1)
        continue_loss = F.binary_cross_entropy_with_logits(continue_logits.float(), continue_target, reduction="none")

        K = min(cfg.imag_last or T, T)
        # With imag_last=0 every posterior is an imagination start, exactly as
        # in the reference implementation. Non-zero values select the tail.
        starts = {
            "deter": posterior["deter"][:, -K:].reshape(B * K, cfg.deter),
            "stoch": posterior["stoch"][:, -K:].reshape(B * K, cfg.stoch, cfg.classes),
        }
        imagined, imagined_actions = self._imagine(starts, cfg.imag_length)
        first = {
            "deter": starts["deter"][:, None],
            "stoch": starts["stoch"][:, None],
        }
        imagined_feature_state = {
            key: torch.cat([first[key], imagined[key]], dim=1)
            for key in ("deter", "stoch")
        }
        if not cfg.ac_grads:
            imagined_feature_state = {key: value.detach() for key, value in imagined_feature_state.items()}
        imagined_feature = self.feature(imagined_feature_state)
        final_logits = self.actor_logits(imagined_feature[:, -1])
        final_action = torch.distributions.Categorical(logits=final_logits).sample()
        imagined_actions = torch.cat([imagined_actions, final_action[:, None]], dim=1)

        imagined_reward = twohot_prediction(self.reward_head(imagined_feature), self.bins)
        imagined_continue = torch.sigmoid(self.continue_head(imagined_feature).squeeze(-1).float())
        policy_logits = self.actor_logits(imagined_feature)
        value_logits = self.value_head(imagined_feature)
        slow_value_logits = self.slow_value_head(imagined_feature)
        value_offset, value_scale = self.value_norm.stats()
        value = twohot_prediction(value_logits, self.bins) * value_scale + value_offset
        slow_value = twohot_prediction(slow_value_logits, self.bins) * value_scale + value_offset
        target_value = slow_value if cfg.slowtar else value
        discount = 1.0 if cfg.contdisc else 1.0 - 1.0 / cfg.horizon
        weight = torch.cumprod(discount * imagined_continue, dim=1) / discount
        returns = self._lambda_return(
            torch.zeros_like(imagined_continue),
            1.0 - imagined_continue,
            imagined_reward,
            target_value,
            target_value,
            discount,
            cfg.lam,
        )
        return_offset, return_scale = self.return_norm(returns, update=update_norms)
        advantage = (returns - target_value[:, :-1]) / return_scale
        advantage_offset, advantage_scale = self.advantage_norm(advantage, update=update_norms)
        normalized_advantage = (advantage - advantage_offset) / advantage_scale
        policy_logprob = F.log_softmax(policy_logits.float(), dim=-1)
        selected_logprob = policy_logprob.gather(-1, imagined_actions.unsqueeze(-1)).squeeze(-1)[:, :-1]
        entropy = _categorical_entropy(policy_logits)[:, :-1]
        policy_loss = weight[:, :-1].detach() * -(
            selected_logprob * normalized_advantage.detach() + cfg.actent * entropy
        )
        value_offset, value_scale = self.value_norm(returns, update=update_norms)
        normalized_return = (returns - value_offset) / value_scale
        padded_return = torch.cat([normalized_return, torch.zeros_like(normalized_return[:, -1:])], dim=1)
        value_loss = weight[:, :-1].detach() * (
            twohot_loss(value_logits, padded_return, self.bins)
            + cfg.slowreg * twohot_loss(value_logits, twohot_prediction(slow_value_logits, self.bins).detach(), self.bins)
        )[:, :-1]
        policy_loss = policy_loss.mean(dim=1).reshape(B, K)
        value_loss = value_loss.mean(dim=1).reshape(B, K)

        replay_value_loss = obs.new_zeros((B, max(0, K - 1)))
        if cfg.repval_loss and K > 1:
            replay_feature = feature[:, -K:] if cfg.repval_grad else feature[:, -K:].detach()
            replay_value_logits = self.value_head(replay_feature)
            replay_slow_logits = self.slow_value_head(replay_feature)
            offset, scale = self.value_norm.stats()
            replay_value = twohot_prediction(replay_value_logits, self.bins) * scale + offset
            replay_slow = twohot_prediction(replay_slow_logits, self.bins) * scale + offset
            replay_target = replay_slow if cfg.slowtar else replay_value
            bootstrap = returns[:, 0].reshape(B, K)
            replay_returns = self._lambda_return(
                batch.is_last[:, -K:],
                batch.is_terminal[:, -K:],
                batch.rewards[:, -K:],
                replay_target,
                bootstrap,
                1.0 - 1.0 / cfg.horizon,
                cfg.lam,
            )
            offset, scale = self.value_norm(replay_returns, update=update_norms)
            normalized = (replay_returns - offset) / scale
            padded = torch.cat([normalized, torch.zeros_like(normalized[:, -1:])], dim=1)
            replay_value_loss = (~batch.is_last[:, -K:]).float()[:, :-1] * (
                twohot_loss(replay_value_logits, padded, self.bins)
                + cfg.slowreg * twohot_loss(
                    replay_value_logits,
                    twohot_prediction(replay_slow_logits, self.bins).detach(),
                    self.bins,
                )
            )[:, :-1]

        valid = batch.valid_mask
        start_valid = valid[:, -K:]
        components = {
            "dyn": self._masked_mean(dyn_loss, valid),
            "rep": self._masked_mean(rep_loss, valid),
            "rec": self._masked_mean(reconstruction_loss, valid),
            "rew": self._masked_mean(reward_loss, valid),
            "con": self._masked_mean(continue_loss, valid),
            "policy": self._masked_mean(policy_loss, start_valid),
            "value": self._masked_mean(value_loss, start_valid),
            "repval": self._masked_mean(replay_value_loss, start_valid[:, :-1]) if K > 1 else obs.new_zeros(()),
        }
        scales = {
            "dyn": cfg.dyn_scale,
            "rep": cfg.rep_scale,
            "rec": cfg.rec_scale,
            "rew": cfg.rew_scale,
            "con": cfg.con_scale,
            "policy": cfg.policy_scale,
            "value": cfg.value_scale,
            "repval": cfg.repval_scale,
        }
        total = sum(components[key] * scales[key] for key in components)
        metrics = {
            **{f"loss_{key}": value.detach() for key, value in components.items()},
            "dyn_entropy": _categorical_entropy(prior_logits, cfg.unimix).sum(dim=-1).mean().detach(),
            "rep_entropy": _categorical_entropy(posterior["logit"], cfg.unimix).sum(dim=-1).mean().detach(),
            "imagined_return": returns.mean().detach(),
            "imagined_reward": imagined_reward.mean().detach(),
            "policy_entropy": entropy.mean().detach(),
        }
        return total.float(), metrics, {key: value.detach() for key, value in final_state.items()}, batch.actions[:, -1].detach()


class DreamerOptimizer:
    """Optax-equivalent AGC -> RMS -> momentum -> learning-rate chain."""

    def __init__(
        self,
        parameters,
        *,
        learning_rate: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup: int = 1000,
    ):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.learning_rate = float(learning_rate)
        self.agc = float(agc)
        self.eps = float(eps)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.warmup = int(warmup)
        self.step_count = 0
        self.rms = [torch.zeros_like(parameter, memory_format=torch.preserve_format) for parameter in self.parameters]
        self.momentum = [torch.zeros_like(parameter, memory_format=torch.preserve_format) for parameter in self.parameters]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> dict[str, float]:
        self.step_count += 1
        gradients = []
        for parameter in self.parameters:
            gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.float()
            if self.agc:
                upper = self.agc * max(1e-3, float(torch.linalg.vector_norm(parameter.detach().float()).item()))
                norm = float(torch.linalg.vector_norm(gradient).item())
                if norm > upper:
                    gradient = gradient * (upper / max(norm, 1e-30))
            gradients.append(gradient)
        beta2_correction = 1.0 - self.beta2 ** self.step_count
        beta1_correction = 1.0 - self.beta1 ** self.step_count
        lr = self.learning_rate
        if self.warmup:
            # Optax evaluates the schedule at count zero for the first update.
            lr *= min(1.0, (self.step_count - 1) / self.warmup)
        updates = []
        for index, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
            self.rms[index].mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
            scaled = gradient / (torch.sqrt(self.rms[index] / beta2_correction) + self.eps)
            self.momentum[index].mul_(self.beta1).add_(scaled, alpha=1.0 - self.beta1)
            update = self.momentum[index] / beta1_correction
            parameter.add_(update.to(parameter.dtype), alpha=-lr)
            updates.append(update)
        grad_norm = math.sqrt(sum(float(value.square().sum().item()) for value in gradients))
        update_rms = math.sqrt(
            sum(float(value.square().sum().item()) for value in updates)
            / max(1, sum(value.numel() for value in updates))
        )
        return {"grad_norm": grad_norm, "update_rms": update_rms, "learning_rate": lr}
