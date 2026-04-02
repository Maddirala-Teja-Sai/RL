"""Experience replay buffer for off-policy algorithms (MADDPG, DDQN)."""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer storing (obs, action, reward, next_obs, done) per agent."""

    def __init__(self, capacity, n_agents, obs_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.n_agents = n_agents
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, n_agents), dtype=np.float32)

    def add(self, obs, actions, rewards, next_obs, dones):
        """Add a transition to the buffer.

        Args:
            obs: (n_agents, obs_dim)
            actions: (n_agents, action_dim)
            rewards: (n_agents,)
            next_obs: (n_agents, obs_dim)
            dones: (n_agents,)
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = dones
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return {
            "obs": torch.FloatTensor(self.obs[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[indices]).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device),
        }

    def __len__(self):
        return self.size
