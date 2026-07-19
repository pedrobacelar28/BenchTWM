"""PPO baseline with tabular actor-critic."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.common import TrainLogger, make_writer, recent_window_mean
from src.envs.factory import make_env
from src.models.mlp import build_actor_critic
from src.utils.seeding import set_seed
from src.utils.torch_utils import get_torch_device


class PPOPolicy:
    def __init__(self, agent):
        self.agent = agent.eval()
        self.device = next(agent.parameters()).device

    def act(self, obs, info, deterministic: bool = True) -> int:
        if not deterministic:
            probs = self.action_probabilities(obs, info, deterministic=False)
            return int(np.random.choice(len(probs), p=probs))
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(info["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.agent.greedy_action(obs_tensor, mask_tensor)
        return int(action.item())

    def action_probabilities(self, obs, info, deterministic: bool = True):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(info["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.agent.greedy_action(obs_tensor, mask_tensor).item()
                probs = torch.zeros(self.agent.policy_logits(obs_tensor, mask_tensor).shape[-1], device=self.device)
                probs[int(action)] = 1.0
            else:
                probs = self.agent.distribution(obs_tensor, mask_tensor).probs.squeeze(0)
        return probs.cpu().numpy()


def train(config: dict, output_dir):
    args = SimpleNamespace(**config)
    set_seed(args.seed)
    device = get_torch_device()
    env = make_env(config, obs_backend=args.obs_backend, seed=args.seed)
    agent = build_actor_critic(
        config,
        obs_dim=env.obs_dim,
        num_states=env.spec.num_states,
        num_actions=env.num_actions,
        obs_backend=args.obs_backend,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    writer = make_writer(bool(args.use_tensorboard), output_dir / "tensorboard")
    logger = TrainLogger(
        gamma=args.gamma,
        success_threshold=args.success_threshold,
        writer=writer,
        algo_name=args.algo,
        max_episodes=args.max_episodes,
        show_progress=bool(getattr(args, "show_progress", True)),
    )

    batch_size = int(args.num_steps)
    minibatch_size = max(1, batch_size // args.num_minibatches)
    obs_buffer = torch.zeros((args.num_steps, env.obs_dim), dtype=torch.float32, device=device)
    action_mask_buffer = torch.zeros((args.num_steps, env.num_actions), dtype=torch.float32, device=device)
    action_buffer = torch.zeros(args.num_steps, dtype=torch.int64, device=device)
    logprob_buffer = torch.zeros(args.num_steps, dtype=torch.float32, device=device)
    reward_buffer = torch.zeros(args.num_steps, dtype=torch.float32, device=device)
    done_buffer = torch.zeros(args.num_steps, dtype=torch.float32, device=device)
    value_buffer = torch.zeros(args.num_steps, dtype=torch.float32, device=device)

    obs, info = env.reset(seed=args.seed)
    done = False
    episode_actions: list[int] = []
    episode_metric_series = [env.transition_metrics(info)]
    episode_return = 0.0
    episode_length = 0
    episode_number = 0
    global_step = 0
    best_score = float("-inf")
    best_state = copy.deepcopy(agent.state_dict())

    while episode_number < args.max_episodes:
        last_step = -1
        for step in range(args.num_steps):
            last_step = step
            obs_buffer[step] = torch.tensor(obs, dtype=torch.float32, device=device)
            action_mask_buffer[step] = torch.tensor(info["action_mask"], dtype=torch.float32, device=device)
            done_buffer[step] = float(done)

            with torch.no_grad():
                action, logprob, _, value = agent.sample_action(obs_buffer[step].unsqueeze(0), action_mask_buffer[step].unsqueeze(0))
            action_buffer[step] = action.squeeze(0)
            logprob_buffer[step] = logprob.squeeze(0)
            value_buffer[step] = value.squeeze(0).squeeze(0)

            next_obs, reward, terminated, truncated, next_info = env.step(int(action.item()))
            reward_buffer[step] = reward
            global_step += 1
            episode_actions.append(int(action.item()))
            episode_metric_series.append(env.transition_metrics(next_info))
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
            obs = next_obs
            info = next_info

            if done:
                outcome = env.episode_metrics(
                    total_reward=episode_return,
                    final_info=info,
                    transition_metrics=episode_metric_series,
                )
                logger.log_episode(
                    episode_number + 1,
                    episode_return,
                    episode_length,
                    global_step,
                    episode_actions,
                    success=bool(outcome["success"]),
                    action_distance=env.action_distance,
                    action_is_abrupt=env.action_is_abrupt,
                    transition_metrics=episode_metric_series,
                    outcome=outcome,
                )
                score = recent_window_mean(logger.returns)
                if score >= best_score:
                    best_score = score
                    best_state = copy.deepcopy(agent.state_dict())
                episode_number += 1
                if episode_number >= args.max_episodes:
                    break
                obs, info = env.reset(seed=args.seed + episode_number)
                done = False
                episode_actions = []
                episode_metric_series = [env.transition_metrics(info)]
                episode_return = 0.0
                episode_length = 0

        with torch.no_grad():
            next_value = agent.get_value(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).squeeze(0)
            advantages = torch.zeros_like(reward_buffer)
            lastgaelam = 0.0
            for t in reversed(range(last_step + 1)):
                if t == last_step:
                    next_non_terminal = 1.0 - float(done)
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - done_buffer[t + 1]
                    next_values = value_buffer[t + 1]
                delta = reward_buffer[t] + args.gamma * next_values * next_non_terminal - value_buffer[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + value_buffer

        b_obs = obs_buffer[: last_step + 1]
        b_action_masks = action_mask_buffer[: last_step + 1]
        b_actions = action_buffer[: last_step + 1]
        b_logprobs = logprob_buffer[: last_step + 1]
        b_advantages = advantages[: last_step + 1]
        b_returns = returns[: last_step + 1]
        b_values = value_buffer[: last_step + 1]

        b_inds = np.arange(last_step + 1)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                mb_inds_t = torch.as_tensor(mb_inds, dtype=torch.int64, device=device)
                newlogprob, entropy, newvalue = agent.evaluate_actions(
                    b_obs[mb_inds_t],
                    b_actions[mb_inds_t],
                    b_action_masks[mb_inds_t],
                )
                logratio = newlogprob - b_logprobs[mb_inds_t]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds_t]
                if args.norm_adv and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds_t]) ** 2
                    v_clipped = b_values[mb_inds_t] + torch.clamp(newvalue - b_values[mb_inds_t], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds_t]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds_t]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                logger.log_extra("policy_loss", float(pg_loss.item()), global_step)
                logger.log_extra("value_loss", float(v_loss.item()), global_step)

        reward_buffer.zero_()
        done_buffer.zero_()
        value_buffer.zero_()

    env.close()
    logger.close()
    if writer is not None:
        writer.close()
    return {
        "metrics": logger.metrics_payload(),
        "checkpoint_last": {"model_state_dict": agent.state_dict()},
        "checkpoint_best": {"model_state_dict": best_state},
    }


def load_policy(config: dict, checkpoint: dict):
    env = make_env(config, obs_backend=config["obs_backend"])
    agent = build_actor_critic(
        config,
        obs_dim=env.obs_dim,
        num_states=env.spec.num_states,
        num_actions=env.num_actions,
        obs_backend=config["obs_backend"],
    ).to(get_torch_device())
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    env.close()
    return PPOPolicy(agent)
