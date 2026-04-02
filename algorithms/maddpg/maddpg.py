"""Multi-Agent Deep Deterministic Policy Gradient (MADDPG).

Centralized training with decentralized execution.
Based on: https://arxiv.org/abs/1706.02275
"""

import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algorithms.base import BaseAlgorithm, device
from algorithms.maddpg.networks import MADDPGActor, MADDPGCritic
from algorithms.maddpg.replay_buffer import ReplayBuffer
from algorithms.maddpg.noise import OUNoise


class MADDPG(BaseAlgorithm):
    """MADDPG: centralized critic, decentralized deterministic actors."""

    def __init__(
        self,
        env,
        eval_config,
        n_agents=8,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=100_000,
        batch_size=256,
        warmup_steps=10000,
        update_interval=100,
        model_dir="models",
        video_dir=None,
        obs_mode="lidar",
        **kwargs,
    ):
        super().__init__(env, eval_config, model_dir, video_dir, obs_mode=obs_mode)
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval

        obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        act_dim = env.action_space(env.possible_agents[0]).shape[0]
        total_obs_dim = obs_dim * n_agents
        total_act_dim = act_dim * n_agents

        # Per-agent actors and centralized critics
        self.actors = [MADDPGActor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.critics = [MADDPGCritic(total_obs_dim, total_act_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a) for a in self.actors]
        self.target_critics = [copy.deepcopy(c) for c in self.critics]

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

        self.noise = [OUNoise(act_dim) for _ in range(n_agents)]
        self.buffer = ReplayBuffer(buffer_size, n_agents, obs_dim, act_dim, device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def learn(self, total_timesteps: int, start_step: int = 0):
        writer = SummaryWriter(log_dir=f"logs/{os.path.basename(self.model_dir)}")
        best_reward = -float("inf")
        global_step = start_step
        episode_count = 0

        print(f"Training MADDPG for {total_timesteps} timesteps")
        if start_step > 0:
            print(f"  Resuming from step {start_step}")
        print(f"  n_agents={self.n_agents}, obs_mode={self.obs_mode}")

        obs_dict = self.env.reset()[0]
        obs_array = self._dict_to_array(obs_dict)
        episode_reward = np.zeros(self.n_agents)

        while global_step < total_timesteps:
            # Get actions
            actions_array = np.zeros((self.n_agents, self.act_dim))
            for i in range(self.n_agents):
                obs_tensor = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = self.actors[i](obs_tensor).cpu().numpy()[0]
                if global_step < self.warmup_steps:
                    action = np.random.uniform(-1, 1, self.act_dim)
                else:
                    action = action + self.noise[i].sample()
                actions_array[i] = np.clip(action, -1, 1)

            # Build action dict
            action_dict = {}
            for i, agent_id in enumerate(self.env.possible_agents):
                if agent_id in self.env.agents:
                    action_dict[agent_id] = actions_array[i]

            if len(action_dict) == 0:
                obs_dict = self.env.reset()[0]
                obs_array = self._dict_to_array(obs_dict)
                episode_reward = np.zeros(self.n_agents)
                for n in self.noise:
                    n.reset()
                continue

            next_obs_dict, rewards_dict, terms, truncs, _ = self.env.step(action_dict)
            rewards = np.array([rewards_dict.get(a, 0.0) for a in self.env.possible_agents])
            dones = np.array([
                float(terms.get(a, False) or truncs.get(a, False))
                for a in self.env.possible_agents
            ])
            next_obs_array = self._dict_to_array(next_obs_dict)

            self.buffer.add(obs_array, actions_array, rewards, next_obs_array, dones)
            episode_reward += rewards
            obs_array = next_obs_array
            global_step += 1

            # Update
            if len(self.buffer) >= self.batch_size and global_step % self.update_interval == 0:
                self._update(writer, global_step)

            # Episode done
            if all(dones):
                episode_count += 1
                mean_ep_reward = episode_reward.mean()
                writer.add_scalar("train/episode_reward", mean_ep_reward, global_step)

                if mean_ep_reward > best_reward and global_step > self.warmup_steps:
                    best_reward = mean_ep_reward
                    self.save_model(os.path.join(self.model_dir, "best_model", "model.pth"))

                if episode_count % 20 == 0:
                    print(f"  Step {global_step} | Episode {episode_count} | "
                          f"Reward: {mean_ep_reward:.3f}")

                obs_dict = self.env.reset()[0]
                obs_array = self._dict_to_array(obs_dict)
                episode_reward = np.zeros(self.n_agents)
                for n in self.noise:
                    n.reset()
                    n.decay()

        writer.close()
        self.save_model(os.path.join(self.model_dir, "final_model.pth"))
        print(f"Training complete. Best reward: {best_reward:.3f}")
        return self

    def _update(self, writer, global_step):
        batch = self.buffer.sample(self.batch_size)
        obs = batch["obs"]           # (B, N, obs_dim)
        actions = batch["actions"]   # (B, N, act_dim)
        rewards = batch["rewards"]   # (B, N)
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        all_obs = obs.reshape(self.batch_size, -1)
        all_actions = actions.reshape(self.batch_size, -1)
        all_next_obs = next_obs.reshape(self.batch_size, -1)

        # Target actions
        target_actions = []
        for i in range(self.n_agents):
            target_actions.append(self.target_actors[i](next_obs[:, i]))
        all_target_actions = torch.cat(target_actions, dim=-1)

        for i in range(self.n_agents):
            # Critic update
            with torch.no_grad():
                target_q = self.target_critics[i](all_next_obs, all_target_actions).squeeze()
                target_value = rewards[:, i] + self.gamma * (1 - dones[:, i]) * target_q

            current_q = self.critics[i](all_obs, all_actions).squeeze()
            critic_loss = F.mse_loss(current_q, target_value)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()

            # Actor update
            curr_actions = []
            for j in range(self.n_agents):
                if j == i:
                    curr_actions.append(self.actors[i](obs[:, i]))
                else:
                    curr_actions.append(actions[:, j])
            all_curr_actions = torch.cat(curr_actions, dim=-1)
            actor_loss = -self.critics[i](all_obs, all_curr_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            # Soft update targets
            self._soft_update(self.target_actors[i], self.actors[i])
            self._soft_update(self.target_critics[i], self.critics[i])

        writer.add_scalar("train/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("train/actor_loss", actor_loss.item(), global_step)

    def _dict_to_array(self, obs_dict):
        return np.array([
            obs_dict.get(a, np.zeros(self.obs_dim))
            for a in self.env.possible_agents
        ])

    def predict(self, observation, deterministic=False):
        """Predict actions for all agents from observation array."""
        if isinstance(observation, dict):
            observation = self._dict_to_array(observation)
        actions = np.zeros((self.n_agents, self.act_dim))
        for i in range(self.n_agents):
            obs_t = torch.FloatTensor(observation[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                actions[i] = self.actors[i](obs_t).cpu().numpy()[0]
        return actions

    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.model_dir, "model.pth")
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "obs_mode": self.obs_mode,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model and optimizer state from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.critics[i].load_state_dict(checkpoint["critics"][i])
            self.target_actors[i] = copy.deepcopy(self.actors[i])
            self.target_critics[i] = copy.deepcopy(self.critics[i])
        
        # Note: MADDPG optimizer state is not saved in original implemention's save_model
        # but we could add it. For now, we just load weights.
        
        try:
            filename = os.path.basename(path)
            if "step_" in filename:
                return int(filename.split("_")[-1].split(".")[0])
        except Exception:
            pass
        return 0

    @classmethod
    def load_model(cls, model_dir: str):
        path = os.path.join(model_dir, "best_model", "model.pth")
        if not os.path.exists(path):
            path = os.path.join(model_dir, "final_model.pth")
        return torch.load(path, map_location=device)
