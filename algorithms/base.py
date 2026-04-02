def set_device(new_device):
    """Update global device for all algorithms."""
    global device
    device = new_device


class BaseAlgorithm(ABC):
    """Base class for all MADRL algorithms."""

    def __init__(self, env, eval_config, model_dir="models", video_dir=None, 
                 device=None, history_length=4, obs_mode="lidar", **kwargs):
        self.device = device if device is not None else globals().get("device", torch.device("cpu"))

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(f"{model_dir}/best_model", exist_ok=True)
        os.makedirs(f"{model_dir}/checkpoints", exist_ok=True)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

    @abstractmethod
    def learn(self, total_timesteps: int):
        """Train the algorithm for total_timesteps."""
        pass

    @abstractmethod
    def predict(self, observation, deterministic=False):
        """Predict action from observation."""
        pass

    @abstractmethod
    def save_model(self, path=None):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, model_dir: str):
        """Load model from disk."""
        pass
