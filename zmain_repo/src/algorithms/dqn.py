"""DQN baseline."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.algorithms.common import TrainLogger, linear_schedule, make_writer, random_admissible_action, recent_window_mean
from src.envs.icu_adapter import ICUSepsisEnvWrapper
from src.models.mlp import TabularQNetwork
from src.utils.replay import ReplayBuffer
from src.utils.seeding import set_seed
from src.utils.torch_utils import get_torch_device


class DQNPolicy:
    def __init__(self, network: TabularQNetwork):
        self.network = network.eval()
        self.device = next(network.parameters()).device

    def act(self, obs, info, deterministic: bool = True) -> int:
        with torch.no_grad():
            q_values = self.network(
                torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(info["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0),
            )
        return int(torch.argmax(q_values, dim=1).item())


def train(config: dict, output_dir):
    args = SimpleNamespace(**config)
    set_seed(args.seed)
    device = get_torch_device()
    env = ICUSepsisEnvWrapper(obs_backend="discrete_id", seed=args.seed)
    q_network = TabularQNetwork(env.num_states, env.num_actions).to(device)
    target_network = TabularQNetwork(env.num_states, env.num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.buffer_size)
    writer = make_writer(bool(args.use_tensorboard), output_dir / "tensorboard")
    logger = TrainLogger(
        gamma=args.gamma,
        success_threshold=args.success_threshold,
        writer=writer,
        algo_name=args.algo,
        max_episodes=args.max_episodes,
        show_progress=bool(getattr(args, "show_progress", True)),
    )

    obs, info = env.reset(seed=args.seed)
    episode_actions: list[int] = []
    episode_return = 0.0
    episode_length = 0
    episode_number = 0
    global_step = 0
    best_score = float("-inf")
    best_state = copy.deepcopy(q_network.state_dict())

    while episode_number < args.max_episodes:
        epsilon = linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.max_episodes), episode_number)
        if random.random() < epsilon:
            action = random_admissible_action(info)
        else:
            with torch.no_grad():
                q_values = q_network(
                    torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
                )
            action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        replay.push(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            action_mask=info["action_mask"],
            next_action_mask=next_info["action_mask"],
            state_id=info["state_id"],
            next_state_id=next_info["state_id"],
        )
        episode_actions.append(action)
        episode_return += reward
        episode_length += 1
        global_step += 1

        if len(replay) >= args.batch_size and global_step > args.learning_starts and global_step % args.train_frequency == 0:
            batch = replay.sample(args.batch_size, device=device)
            with torch.no_grad():
                target_max = target_network(batch.next_obs, batch.next_action_mask).max(dim=1).values
                td_target = batch.reward + args.gamma * target_max * (1.0 - batch.done)
            old_val = q_network(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(old_val, td_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.log_extra("td_loss", float(loss.item()), global_step)

        if global_step % args.target_network_frequency == 0:
            for target_param, source_param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(args.tau * source_param.data + (1.0 - args.tau) * target_param.data)

        obs = next_obs
        info = next_info

        if done:
            logger.log_episode(episode_number + 1, episode_return, episode_length, global_step, episode_actions)
            score = recent_window_mean(logger.returns)
            if score >= best_score:
                best_score = score
                best_state = copy.deepcopy(q_network.state_dict())
            episode_number += 1
            obs, info = env.reset(seed=args.seed + episode_number)
            episode_actions = []
            episode_return = 0.0
            episode_length = 0

    env.close()
    logger.close()
    if writer is not None:
        writer.close()
    return {
        "metrics": logger.metrics_payload(),
        "checkpoint_last": {"model_state_dict": q_network.state_dict()},
        "checkpoint_best": {"model_state_dict": best_state},
    }


def load_policy(config: dict, checkpoint: dict):
    env = ICUSepsisEnvWrapper(obs_backend="discrete_id")
    network = TabularQNetwork(env.num_states, env.num_actions).to(get_torch_device())
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    env.close()
    return DQNPolicy(network)
