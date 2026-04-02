"""MAPPO with the original temporal LiDAR encoder from navigation-mappo-rl."""

import copy
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

from algorithms.base import BaseAlgorithm, device
from algorithms.mappo.distributions import DiagGaussianDistribution
from algorithms.mappo.rollout_buffer import RolloutBuffer
from networks.original_observation_encoder import OriginalObservationEncoder


class Actor(nn.Module):
    def __init__(
        self,
        raw_obs_dim: int,
        action_dim: int,
        agent_states_dim: int,
        lidar_dim: int,
        history_length: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = OriginalObservationEncoder(
            raw_obs_dim=raw_obs_dim,
            agent_states_dim=agent_states_dim,
            lidar_dim=lidar_dim,
            history_length=history_length,
            objects=3,
            features_dim=hidden_dim,
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.dist = DiagGaussianDistribution(action_dim)

    def forward(self, obs):
        features = self.encoder(obs)
        mean = self.mean_head(features)
        return self.dist.proba_distribution(mean, self.log_std)

    def get_action(self, obs, deterministic=False):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, obs, actions):
        dist = self.forward(obs)
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    def __init__(
        self,
        raw_obs_dim: int,
        n_agents: int,
        agent_states_dim: int,
        lidar_dim: int,
        history_length: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.encoder = OriginalObservationEncoder(
            raw_obs_dim=raw_obs_dim,
            agent_states_dim=agent_states_dim,
            lidar_dim=lidar_dim,
            history_length=history_length,
            objects=3,
            features_dim=hidden_dim,
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs):
        if joint_obs.dim() == 4:
            joint_obs = joint_obs.squeeze(1)
        if joint_obs.dim() == 2:
            joint_obs = joint_obs.unsqueeze(0)

        batch_size, n_agents, obs_dim = joint_obs.shape
        encoded = self.encoder(joint_obs.reshape(batch_size * n_agents, obs_dim))
        encoded = encoded.reshape(batch_size, n_agents, -1)

        query = self.query(encoded)
        key = self.key(encoded)
        value = self.value(encoded)
        attended, _ = self.attention(query, key, value)
        pooled = attended.mean(dim=1)
        return self.value_head(pooled)


class MAPPO(BaseAlgorithm):
    def __init__(
        self,
        env,
        eval_config,
        n_agents=8,
        n_steps=256,
        n_epochs=4,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lr=5e-4,
        history_length=4,
        hidden_dim=256,
        ema_decay=0.0,
        model_dir="models",
        video_dir=None,
        obs_mode="lidar",
        **kwargs,
    ):
        super().__init__(env, eval_config, model_dir, video_dir, obs_mode=obs_mode)
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.history_length = history_length
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        self.raw_obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        self.agent_states_dim = env.agent_states_dim
        self.lidar_dim = env.config.num_rays
        self.stacked_obs_dim = self.raw_obs_dim * self.history_length
        self.act_dim = env.action_space(env.possible_agents[0]).shape[0]

        self.actor = Actor(
            raw_obs_dim=self.raw_obs_dim,
            action_dim=self.act_dim,
            agent_states_dim=self.agent_states_dim,
            lidar_dim=self.lidar_dim,
            history_length=self.history_length,
            hidden_dim=self.hidden_dim,
        ).to(device)
        self.critic = Critic(
            raw_obs_dim=self.raw_obs_dim,
            n_agents=self.n_agents,
            agent_states_dim=self.agent_states_dim,
            lidar_dim=self.lidar_dim,
            history_length=self.history_length,
            hidden_dim=self.hidden_dim,
        ).to(device)
        self.ema_actor = None
        if self.ema_decay > 0:
            self.ema_actor = copy.deepcopy(self.actor).to(device)
            self.ema_actor.eval()
            for param in self.ema_actor.parameters():
                param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )

        stacked_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacked_obs_dim,),
            dtype=np.float32,
        )
        self.buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=stacked_obs_space,
            action_space=env.action_space(env.possible_agents[0]),
            n_agents=n_agents,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_agents,
        )

    def _init_histories(self, obs_dict):
        histories = {}
        zero_obs = np.zeros(self.raw_obs_dim, dtype=np.float32)
        for agent_id in self.env.possible_agents:
            obs = np.array(obs_dict.get(agent_id, zero_obs), dtype=np.float32)
            history = deque(maxlen=self.history_length)
            for _ in range(self.history_length):
                history.append(obs.copy())
            histories[agent_id] = history
        return histories

    def _advance_histories(self, histories, obs_dict):
        zero_obs = np.zeros(self.raw_obs_dim, dtype=np.float32)
        for agent_id in self.env.possible_agents:
            obs = np.array(obs_dict.get(agent_id, zero_obs), dtype=np.float32)
            histories[agent_id].append(obs)

    def _stack_histories(self, histories):
        return np.stack(
            [
                np.concatenate(list(histories[agent_id]), axis=0)
                for agent_id in self.env.possible_agents
            ]
        ).astype(np.float32)

    def _update_ema_actor(self):
        if self.ema_actor is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_actor.parameters(), self.actor.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)
            for ema_buffer, buffer in zip(self.ema_actor.buffers(), self.actor.buffers()):
                ema_buffer.copy_(buffer)

    def learn(self, total_timesteps: int):
        writer = SummaryWriter(log_dir=f"logs/{os.path.basename(self.model_dir)}")
        num_updates = max(1, total_timesteps // (self.n_steps * self.n_agents))
        best_reward = -float("inf")
        global_step = 0

        print(f"Training MAPPO for {total_timesteps} timesteps ({num_updates} updates)")
        print(
            f"  n_agents={self.n_agents}, history_length={self.history_length}, "
            f"obs_mode={self.obs_mode}"
        )

        for update in range(1, num_updates + 1):
            self.buffer.reset()
            episode_rewards = []

            obs_dict = self.env.reset()[0]
            histories = self._init_histories(obs_dict)
            dones = np.zeros(self.n_agents, dtype=np.float32)

            for _ in range(self.n_steps):
                stacked_obs = self._stack_histories(histories)
                obs_tensor = torch.as_tensor(stacked_obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    actions, log_probs = self.actor.get_action(obs_tensor)
                    values = self.critic(obs_tensor.unsqueeze(0)).squeeze()
                    if values.dim() == 0:
                        values = values.unsqueeze(0).expand(self.n_agents)
                    else:
                        values = values.expand(self.n_agents)

                actions_np = np.clip(actions.cpu().numpy(), -1, 1)
                action_dict = {}
                for idx, agent_id in enumerate(self.env.possible_agents):
                    if agent_id in self.env.agents:
                        action_dict[agent_id] = actions_np[idx]

                if not action_dict:
                    obs_dict = self.env.reset()[0]
                    histories = self._init_histories(obs_dict)
                    dones = np.zeros(self.n_agents, dtype=np.float32)
                    continue

                next_obs_dict, rewards_dict, terms, truncs, _ = self.env.step(action_dict)
                rewards = np.array(
                    [rewards_dict.get(agent_id, 0.0) for agent_id in self.env.possible_agents],
                    dtype=np.float32,
                )
                episode_rewards.append(rewards.mean())

                new_dones = np.array(
                    [
                        float(terms.get(agent_id, False) or truncs.get(agent_id, False))
                        for agent_id in self.env.possible_agents
                    ],
                    dtype=np.float32,
                )

                self.buffer.add(
                    stacked_obs,
                    actions_np,
                    rewards,
                    dones,
                    values,
                    log_probs,
                    stacked_obs.reshape(1, self.n_agents, -1),
                )

                self._advance_histories(histories, next_obs_dict)
                dones = new_dones
                global_step += self.n_agents

                if all(new_dones):
                    obs_dict = self.env.reset()[0]
                    histories = self._init_histories(obs_dict)
                    dones = np.zeros(self.n_agents, dtype=np.float32)

            with torch.no_grad():
                last_stacked_obs = self._stack_histories(histories)
                last_values = self.critic(
                    torch.as_tensor(last_stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)
                ).squeeze()
                if last_values.dim() == 0:
                    last_values = last_values.unsqueeze(0).expand(self.n_agents)
                else:
                    last_values = last_values.expand(self.n_agents)

            self.buffer.compute_returns_and_advantage(last_values, dones)

            policy_losses = []
            value_losses = []
            entropy_losses = []
            for _ in range(self.n_epochs):
                for rollout_data in self.buffer.get(self.batch_size):
                    new_log_probs, entropy = self.actor.evaluate_actions(
                        rollout_data.observations,
                        rollout_data.actions,
                    )
                    new_values = self.critic(rollout_data.states).squeeze()

                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    ratio = (new_log_probs - rollout_data.old_log_prob).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.functional.mse_loss(new_values, rollout_data.returns)
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self._update_ema_actor()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())

            mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            writer.add_scalar("train/mean_reward", mean_reward, global_step)
            writer.add_scalar("train/policy_loss", np.mean(policy_losses), global_step)
            writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
            writer.add_scalar("train/entropy", -np.mean(entropy_losses), global_step)

            if update % 10 == 0:
                print(
                    f"  Update {update}/{num_updates} | Step {global_step} | "
                    f"Reward: {mean_reward:.3f} | PL: {np.mean(policy_losses):.4f} | "
                    f"VL: {np.mean(value_losses):.4f}"
                )

            if mean_reward > best_reward:
                best_reward = mean_reward
                self.save_model(os.path.join(self.model_dir, "best_model", "model.pth"))

            if update % 50 == 0:
                self.save_model(
                    os.path.join(self.model_dir, "checkpoints", f"model_step_{global_step}.pth")
                )

        writer.close()
        self.save_model(os.path.join(self.model_dir, "final_model.pth"))
        print(f"Training complete. Best reward: {best_reward:.3f}")
        return self

    def predict(self, observation, deterministic=False):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)
        return action.cpu().numpy()

    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.model_dir, "model.pth")
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "ema_actor": self.ema_actor.state_dict() if self.ema_actor is not None else None,
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_mode": self.obs_mode,
                "ema_decay": self.ema_decay,
                "actor_config": {
                    "raw_obs_dim": self.raw_obs_dim,
                    "action_dim": self.act_dim,
                    "agent_states_dim": self.agent_states_dim,
                    "lidar_dim": self.lidar_dim,
                    "history_length": self.history_length,
                    "hidden_dim": self.hidden_dim,
                },
            },
            path,
        )

    @classmethod
    def load_model(cls, model_dir: str):
        path = os.path.join(model_dir, "best_model", "model.pth")
        if not os.path.exists(path):
            path = os.path.join(model_dir, "final_model.pth")
        return torch.load(path, map_location=device)
