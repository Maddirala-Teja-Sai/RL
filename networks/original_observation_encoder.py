"""Original LiDAR history encoder adapted from navigation-mappo-rl."""

import torch
import torch.nn as nn


class OriginalObservationEncoder(nn.Module):
    """Encode stacked agent-state and LiDAR history with a 1D CNN backbone."""

    def __init__(
        self,
        raw_obs_dim: int,
        agent_states_dim: int,
        lidar_dim: int,
        history_length: int,
        objects: int = 3,
        features_dim: int = 256,
    ):
        super().__init__()
        self.history_length = history_length
        self.agent_states_dim = agent_states_dim
        self.lidar_dim = lidar_dim

        channels = history_length * objects
        self.cnn_1d = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.agent_network = nn.Sequential(
            nn.Linear(agent_states_dim * history_length, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_obs = torch.zeros(1, channels, lidar_dim)
            n_flatten = self.cnn_1d(sample_obs).shape[1]

        self.dense_layer = nn.Sequential(
            nn.Linear(n_flatten + 128, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        observations = observations.reshape(batch_size, self.history_length, -1)

        lidar_observations = observations[:, :, self.agent_states_dim :]
        lidar_observations = lidar_observations.reshape(
            batch_size, self.history_length, 3, self.lidar_dim
        )
        lidar_observations = lidar_observations.reshape(
            batch_size, self.history_length * 3, self.lidar_dim
        )
        agent_observations = observations[:, :, : self.agent_states_dim].flatten(start_dim=1)

        lidar_encoded = self.cnn_1d(lidar_observations)
        agent_encoded = self.agent_network(agent_observations)
        return self.dense_layer(torch.cat([lidar_encoded, agent_encoded], dim=1))
