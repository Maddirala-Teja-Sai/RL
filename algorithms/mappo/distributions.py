import torch
import torch.nn as nn
from torch.distributions import Normal


class DiagGaussianDistribution:
    """Diagonal Gaussian distribution for continuous action spaces."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> "DiagGaussianDistribution":
        log_std = torch.clamp(log_std, -20, 2)
        self.distribution = Normal(mean_actions, log_std.exp())
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def sample(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized.")
        return self.distribution.sample()

    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized.")
        return self.distribution.entropy().sum(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized.")
        return self.distribution.mean
