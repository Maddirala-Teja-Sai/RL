from nav.environment import Environment
from nav.config_models import EnvConfig
import numpy as np
import traceback
import yaml
import sys

try:
    print("Loading marl_mixed.yaml...")
    with open("configs/marl_mixed.yaml", "r") as f:
        raw = yaml.safe_load(f)
    config = EnvConfig(**raw)

    print("Creating environment...")
    env = Environment(config)
    
    print("Resetting...")
    obs, info = env.reset()
    print("Reset complete. Obs keys:", list(obs.keys()))
    
    print("Stepping...")
    # Random actions
    actions = {agent: np.random.uniform(-1, 1, 2).astype(np.float32) for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    print("Step complete.")

except Exception:
    with open("debug_error.log", "w") as f:
        traceback.print_exc(file=f)
    traceback.print_exc()
