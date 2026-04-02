"""Unified training script for MADRL Navigation.

Supports: MAPPO, MADDPG
Observation modes: lidar, vision, bilinear
Environments: marl_static, marl_moving, marl_mixed

Usage:
    python train.py --algo mappo --config configs/marl_static.yaml --model-id my_model --timesteps 2000000
"""

import argparse
import os
import shutil
import yaml
import numpy as np
import torch

from nav.environment import Environment
from nav.config_models import EnvConfig


def parse_args():
    parser = argparse.ArgumentParser(description="MADRL Navigation Training")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["mappo", "maddpg"],
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
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps (default: 2000000)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=5,
        help="Number of agents",
    )
    parser.add_argument(
        "--drive-path",
        type=str,
        default=None,
        help="Google Drive path to sync results (e.g., /content/drive/MyDrive/RL_Models)",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for actor networks (e.g. 0.995)",
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
        help="Path to a specific checkpoint to resume from",
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
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def main():
    args = parse_args()

    # Paths
    # We use absolute paths to prevent any "missing file" issues in Colab
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, "models", args.model_id)
    video_dir = os.path.join(cwd, "videos", args.model_id)
    log_dir = os.path.join(cwd, "logs", args.model_id)

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("MADRL Navigation Training")
    print("=" * 60)
    print(f"  Algorithm:   {args.algo}")
    print(f"  Config:      {args.config}")
    print(f"  Obs Mode:    {args.obs_mode}")
    print(f"  Model ID:    {args.model_id}")
    print(f"  Timesteps:   {args.timesteps:,}")
    print(f"  EMA Decay:   {args.ema_decay}")
    print(f"  History:     {args.history_length}")
    print(f"  Save Dir:    {model_dir}")
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
    from algorithms.base import set_device
    device_obj = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    set_device(device_obj)
    print(f"  Training on: {device_obj.type.upper()}")
    
    # Create algorithm
    AlgoClass = get_algorithm(args.algo)

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

    # Resume or Checkpoint Loading
    start_step = 0
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint
        if not checkpoint_path and args.resume:
            # First, check for the specific 'checkpoints' folder
            checkpoint_dir = os.path.join(model_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
                if files:
                    # Sort files by step number: model_step_500000.pth
                    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    checkpoint_path = os.path.join(checkpoint_dir, files[-1])
            
            # If no checkpoints, try the best model
            if not checkpoint_path:
                best_path = os.path.join(model_dir, "best_model", "model.pth")
                if os.path.exists(best_path):
                    checkpoint_path = best_path
                
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n🔄 RESUMING FROM CHECKPOINT: {checkpoint_path}")
            start_step = learner.load_checkpoint(checkpoint_path)
            print(f"📈 Resuming from Step: {start_step:,}")
        else:
            print("\n⚠️ No valid checkpoint found. Starting training from scratch (Step 0).")

    # Train
    # Note: args.timesteps is the TARGET TOTAL step. 
    # If start_step=5M and args.timesteps=10M, it will train for 5M more.
    learner.learn(total_timesteps=args.timesteps, start_step=start_step)

    # Final Verification
    final_path = os.path.join(model_dir, "final_model.pth")
    if os.path.exists(final_path):
        print(f"\nSUCCESS: Model saved and verified at: {final_path}")
    else:
        print(f"\nWARNING: Could not verify final_model.pth at {final_path}")

    print(f"TensorBoard logs: logs/{args.model_id}")

    # Google Drive Sync
    if args.drive_path:
        print(f"\n📂 SYNCING TO GOOGLE DRIVE: {args.drive_path}")
        dest_base = os.path.join(args.drive_path, args.model_id)
        os.makedirs(dest_base, exist_ok=True)
        
        # Sync Models
        dest_models = os.path.join(dest_base, "models")
        if os.path.exists(dest_models): shutil.rmtree(dest_models)
        shutil.copytree(model_dir, dest_models)
        
        # Sync Logs
        dest_logs = os.path.join(dest_base, "logs")
        curr_logs = os.path.join("logs", args.model_id)
        if os.path.exists(curr_logs):
            if os.path.exists(dest_logs): shutil.rmtree(dest_logs)
            shutil.copytree(curr_logs, dest_logs)
            
        print(f"✅ Sync complete! Results saved to Google Drive.")


if __name__ == "__main__":
    main()
