import numpy as np
import torch as th
from gymnasium import spaces
from typing import Optional, Generator, NamedTuple, Union


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    states: th.Tensor


class RolloutBuffer:
    """On-policy rollout buffer for MAPPO with centralized state support."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_agents: int,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_agents = n_agents
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.pos = 0
        self.full = False
        self.generator_ready = False
        self.reset()

    def reset(self):
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.states = np.zeros(
            (self.buffer_size, self.n_envs // self.n_agents, self.n_agents) + self.obs_shape,
            dtype=np.float32,
        )
        self.pos = 0
        self.full = False
        self.generator_ready = False

    @property
    def size(self):
        return self.buffer_size if self.full else self.pos

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray):
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self, obs, action, reward, episode_start, value, log_prob, state):
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().flatten()
        self.states[self.pos] = np.array(state).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None):
        assert self.full or self.pos > 0
        if not self.generator_ready:
            for key in ["observations", "actions", "values", "log_probs", 
                        "advantages", "returns", "states"]:
                arr = getattr(self, key)
                setattr(self, key, self.swap_and_flatten(arr))
            self.generator_ready = True

        total = self.buffer_size * self.n_envs
        if batch_size is None:
            batch_size = total
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        if len(shape) < 2:
            return arr
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def _get_samples(self, batch_inds: np.ndarray):
        n_actual = self.buffer_size * self.n_envs
        state_inds = batch_inds // self.n_agents
        state_inds = np.clip(state_inds, 0, len(self.states) - 1)
        return RolloutBufferSamples(
            observations=self.to_torch(self.observations[batch_inds]),
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            states=self.to_torch(self.states[state_inds]),
        )

    def to_torch(self, array: np.ndarray, copy: bool = True):
        if copy:
            return th.tensor(array, device=self.device, dtype=th.float32)
        return th.as_tensor(array, device=self.device)
