"""Inference script for MADRL Navigation.

Loads a trained model and runs agents in the environment with:
- Live visualization (human mode) via Arcade window
- Video recording to movies/ directory

Usage:
    uv run inference.py models/my_model configs/marl_static.yaml
    uv run inference.py models/my_model configs/marl_static.yaml --mode video
    uv run inference.py models/my_model configs/marl_static.yaml --episodes 5
"""

import argparse
from collections import deque
import os
import sys
import time
import numpy as np
import yaml
import torch

from nav.environment import Environment
from nav.config_models import EnvConfig


def parse_args():
    parser = argparse.ArgumentParser(description="MADRL Navigation Inference")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory (e.g., models/my_model)",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to environment YAML config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "video"],
        help="Render mode: human (live window) or video (save .mp4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="auto",
        choices=["auto", "online", "ema"],
        help="Which MAPPO actor weights to use for inference",
    )
    return parser.parse_args()


def load_metadata(model_dir):
    """Load training metadata from model directory."""
    meta_path = os.path.join(model_dir, "metadata.yaml")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def load_model_weights(model_dir, algo_name, env, n_agents, policy_name="auto", metadata=None):
    """Load trained model weights and return a predictor function."""
    # Find model file
    model_path = os.path.join(model_dir, "best_model", "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: No model found in {model_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]

    if algo_name == "mappo":
        from algorithms.mappo.mappo_lidar_history import Actor
        act_dim = env.action_space(env.possible_agents[0]).shape[0]
        actor_config = checkpoint.get("actor_config", {})
        actor = Actor(
            raw_obs_dim=actor_config.get("raw_obs_dim", obs_dim),
            action_dim=actor_config.get("action_dim", act_dim),
            agent_states_dim=actor_config.get("agent_states_dim", env.agent_states_dim),
            lidar_dim=actor_config.get("lidar_dim", env.lidar_dim),
            history_length=actor_config.get("history_length", metadata.get("history_length", 4) if metadata else 4),
            hidden_dim=actor_config.get("hidden_dim", 256),
        ).to(device)
        requested_policy = policy_name
        if requested_policy == "auto":
            requested_policy = (metadata or {}).get("default_inference_policy", "online")

        actor_state = checkpoint["actor"]
        if requested_policy == "ema":
            if checkpoint.get("ema_actor") is None:
                raise ValueError(
                    "This MAPPO checkpoint does not include EMA weights. "
                    "Run inference with --policy online or retrain with --ema-decay > 0."
                )
            actor_state = checkpoint["ema_actor"]

        actor.load_state_dict(actor_state)
        actor.eval()
        history_length = actor_config.get("history_length", metadata.get("history_length", 4) if metadata else 4)
        zero_obs = np.zeros(obs_dim, dtype=np.float32)
        histories = {
            agent_id: deque([zero_obs.copy() for _ in range(history_length)], maxlen=history_length)
            for agent_id in env.possible_agents
        }

        def predict(obs_dict):
            actions = {}
            for agent_id in env.possible_agents:
                obs = np.array(obs_dict.get(agent_id, zero_obs), dtype=np.float32)
                histories[agent_id].append(obs)

            for agent_id, obs in obs_dict.items():
                stacked_obs = np.concatenate(list(histories[agent_id]), axis=0)
                obs_t = torch.FloatTensor(stacked_obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _ = actor.get_action(obs_t, deterministic=True)
                actions[agent_id] = np.clip(action.cpu().numpy()[0], -1, 1)
            return actions
        return predict

    elif algo_name == "maddpg":
        from algorithms.maddpg.networks import MADDPGActor
        act_dim = env.action_space(env.possible_agents[0]).shape[0]
        actors = []
        for i, state_dict in enumerate(checkpoint["actors"]):
            actor = MADDPGActor(obs_dim, act_dim).to(device)
            actor.load_state_dict(state_dict)
            actor.eval()
            actors.append(actor)

        def predict(obs_dict):
            actions = {}
            for i, agent_id in enumerate(env.possible_agents):
                if agent_id in obs_dict:
                    obs_t = torch.FloatTensor(obs_dict[agent_id]).unsqueeze(0).to(device)
                    actor_idx = min(i, len(actors) - 1)
                    with torch.no_grad():
                        action = actors[actor_idx](obs_t).cpu().numpy()[0]
                    actions[agent_id] = np.clip(action, -1, 1)
            return actions
        return predict


    elif algo_name == "multimodal":
        from algorithms.multimodal.multimodal_drl import MultimodalActor
        act_dim = env.action_space(env.possible_agents[0]).shape[0]
        actor = MultimodalActor(obs_dim, act_dim).to(device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

        def predict(obs_dict):
            actions = {}
            for agent_id, obs in obs_dict.items():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                dummy_img = torch.zeros(1, 3, 128, 128).to(device)
                with torch.no_grad():
                    action, _ = actor.get_action(obs_t, dummy_img, deterministic=True)
                actions[agent_id] = np.clip(action.cpu().numpy()[0], -1, 1)
            return actions
        return predict

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_inference(env, predict_fn, n_episodes, mode, model_dir):
    """Run inference episodes with visualization or video recording."""
    os.makedirs("movies", exist_ok=True)

    for ep in range(1, n_episodes + 1):
        print(f"\n--- Episode {ep}/{n_episodes} ---")
        obs_dict = env.reset()[0]

        total_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
        frames = []
        step = 0

        while env.agents:
            actions = predict_fn(obs_dict)
            obs_dict, rewards, terms, truncs, infos = env.step(actions)

            for agent_id, r in rewards.items():
                total_rewards[agent_id] = total_rewards.get(agent_id, 0) + r

            if mode == "human":
                env.render()
                time.sleep(0.02)  # ~50 FPS

            step += 1

            # Check if all done
            all_done = all(
                terms.get(a, False) or truncs.get(a, False)
                for a in env.possible_agents
            )
            if all_done:
                break

        mean_reward = np.mean(list(total_rewards.values()))
        goals_reached = sum(1 for a in env.agents_dict.values() if a.goal_reached)
        print(f"  Steps: {step} | Mean Reward: {mean_reward:.3f} | "
              f"Goals: {goals_reached}/{len(env.possible_agents)}")

    print(f"\n{'='*40}")
    print("Inference complete!")


def main():
    args = parse_args()

    # Load metadata
    meta = load_metadata(args.model_dir)
    algo_name = meta.get("algo", "mappo")
    n_agents = meta.get("n_agents", 8)

    print("=" * 60)
    print("MADRL Navigation Inference")
    print("=" * 60)
    print(f"  Model:     {args.model_dir}")
    print(f"  Algorithm: {algo_name}")
    print(f"  Config:    {args.config}")
    print(f"  Mode:      {args.mode}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Policy:    {args.policy}")
    print("=" * 60)

    # Load config and create environment
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    config = EnvConfig(**raw)

    render_mode = "human" if args.mode == "human" else None
    env = Environment(config=config, render_mode=render_mode)

    # Load model
    predict_fn = load_model_weights(
        args.model_dir,
        algo_name,
        env,
        n_agents,
        policy_name=args.policy,
        metadata=meta,
    )

    # Run
    run_inference(env, predict_fn, args.episodes, args.mode, args.model_dir)


if __name__ == "__main__":
    main()
