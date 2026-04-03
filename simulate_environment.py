import sys
import os
import yaml
import numpy as np
import imageio
from nav.environment import Environment
from nav.config_models import EnvConfig

def simulate_and_save(config_path, output_path="env_preview.gif", num_steps=100):
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = EnvConfig(**config_dict)
    
    print("Initializing environment in headless mode...")
    # render_mode='rgb_array' triggers headless mode in our SimulationWindow
    env = Environment(config=config, render_mode="rgb_array")
    
    obs, info = env.reset()
    frames = []
    
    print(f"Simulating {num_steps} steps...")
    for i in range(num_steps):
        # Use random actions for the preview
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Capture frame
        env.render()
        frame = env.window.get_rgb_array()
        if frame is not None:
            frames.append(frame)
        
        if i % 10 == 0:
            print(f"  Step {i}/{num_steps}")
            
        if any(terminations.values()) or any(truncations.values()):
            break
            
    print(f"Saving {len(frames)} frames to {output_path}...")
    imageio.mimsave(output_path, frames, fps=10)
    print("Done!")

if __name__ == "__main__":
    cfg = "configs/big_environment.yaml"
    if len(sys.argv) > 1:
        cfg = sys.argv[1]
    
    simulate_and_save(cfg)
