"""Unified training script for MADRL Navigation.

Supports: MAPPO, MADDPG, Double DQN, Multimodal DRL
Observation modes: lidar, vision, bilinear
Environments: marl_static, marl_moving, marl_mixed

Usage:
    uv run train.py --algo mappo --config configs/marl_static.yaml --obs-mode lidar --model-id my_model
    uv run train.py --algo ddqn --config configs/marl_moving.yaml --obs-mode vision --model-id test 10000
"""

import argparse
import os
import shutil
import yaml
import __main__

from nav.environment import Environment
from nav.config_models import EnvConfig


def parse_args():
    parser = argparse.ArgumentParser(description="MADRL Navigation Training")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["mappo", "maddpg", "ddqn", "multimodal"],
        help="Algorithm to train with",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to environment YAML config",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="lidar",
        choices=["lidar", "vision", "bilinear"],
        help="Observation mode",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Unique model identifier for saving",
    )
    parser.add_argument(
        "timesteps",
        type=int,
        nargs="?",
        default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=8,
        help="Number of agents",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for MAPPO actor averaging (0 disables EMA)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=4,
        help="Temporal history length for MAPPO's LiDAR encoder",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force use of CPU even if CUDA is available",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in the model directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file to resume from (optional)",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load environment configuration from YAML."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return EnvConfig(**raw)


def create_env(config):
    """Create PettingZoo environment from config."""
    env = Environment(config=config)
    return env


def get_algorithm(algo_name):
    """Import and return the algorithm class."""
    if algo_name == "mappo":
        from algorithms.mappo import MAPPO
        return MAPPO
    elif algo_name == "maddpg":
        from algorithms.maddpg import MADDPG
        return MADDPG
    elif algo_name == "ddqn":
        from algorithms.ddqn import DoubleDQN
        return DoubleDQN
    elif algo_name == "multimodal":
        from algorithms.multimodal import MultimodalDRL
        return MultimodalDRL
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def main():
    args = parse_args()

    print("=" * 60)
    print("MADRL Navigation Training")
    print("=" * 60)
    print(f"  Algorithm:   {args.algo}")
    print(f"  Config:      {args.config}")
    print(f"  Obs Mode:    {args.obs_mode}")
    print(f"  Model ID:    {args.model_id}")
    print(f"  Timesteps:   {args.timesteps:,}")
    print(f"  Agents:      {args.n_agents}")
    print(f"  EMA Decay:   {args.ema_decay}")
    print(f"  History:     {args.history_length}")
    print("=" * 60)

    # Load config and create environment
    config = load_config(args.config)
    env = create_env(config)
    n_agents = env.n_agents
    # Copy config to model directory for reference
    shutil.copy2(args.config, os.path.join(model_dir, "env.yaml"))

    # Save training metadata
    with open(os.path.join(model_dir, "metadata.yaml"), "w") as f:
        yaml.dump({
            "algo": args.algo,
            "config": args.config,
            "obs_mode": args.obs_mode,
            "model_id": args.model_id,
            "timesteps": args.timesteps,
            "n_agents": n_agents,
            "ema_decay": args.ema_decay,
            "history_length": args.history_length,
            "default_inference_policy": (
                "ema" if args.algo == "mappo" and args.ema_decay > 0 else "online"
            ),
        }, f)

    # Device selection
    import torch
    from algorithms.base import set_device
    device_obj = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    set_device(device_obj)
    print(f"  Training on: {device_obj.type.upper()}")
    
    # Create algorithm
    AlgoClass = get_algorithm(args.algo)

    if args.algo == "multimodal":
        # Multimodal always uses bilinear internally
        learner = AlgoClass(
            env=env,
            eval_config=config,
            n_agents=n_agents,
            model_dir=model_dir,
            video_dir=video_dir,
        )
    else:
        learner = AlgoClass(
            env=env,
            eval_config=config,
            n_agents=n_agents,
            model_dir=model_dir,
            video_dir=video_dir,
            obs_mode=args.obs_mode,
            ema_decay=args.ema_decay,
            history_length=args.history_length,
        )

    # Resume if requested
    start_step = 0
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint
        if not checkpoint_path and args.resume:
            # Find latest checkpoint
            checkpoint_dir = os.path.join(model_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
                if checkpoints:
                    # Sort by step number: model_step_X.pth
                    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
            
            # If no checkpoint in checkpoints/, try best_model/model.pth
            if not checkpoint_path:
                best_path = os.path.join(model_dir, "best_model", "model.pth")
                if os.path.exists(best_path):
                    checkpoint_path = best_path

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            start_step = learner.load_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found to resume from. Starting from scratch.")

    # Train
    learner.learn(total_timesteps=args.timesteps, start_step=start_step)

    print(f"\nModels saved to: {model_dir}")
    print(f"TensorBoard logs: logs/{args.model_id}")
    print(f"Run: tensorboard --logdir logs/")
if __name__ == "__main__":
    main()
