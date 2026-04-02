# MADRL Navigation — Multi-Agent Deep Reinforcement Learning

Multi-algorithm, multi-observation navigation project. 8 agents navigate to fixed goals while avoiding obstacles, spawning randomly within a fixed boundary each episode.

## Algorithms

| Algorithm | Type | Actions |
|-----------|------|---------|
| **MAPPO** | On-policy (PPO) | Continuous |
| **MADDPG** | Off-policy (DDPG) | Continuous |
| **Double DQN** | Off-policy (DQN) | 25 discrete |
| **Multimodal DRL** | On-policy (PPO) | Continuous |

## Observation Modes

| Mode | Description |
|------|-------------|
| `lidar` | LIDAR rays + agent state vector |
| `vision` | 128×128 top-down Faster R-CNN features |
| `bilinear` | LIDAR × Vision bilinear fusion |

## Environments

| Config | Difficulty | Obstacles |
|--------|-----------|-----------|
| `marl_static.yaml` | Easy | Static only |
| `marl_moving.yaml` | Medium | Moving only |
| `marl_mixed.yaml` | Hard | Static + Moving |

## Setup

```bash
uv sync
```

## Training

```bash
# Train MAPPO with LIDAR on static environment (100K steps default)
uv run train.py --algo mappo --config configs/marl_static.yaml --obs-mode lidar --model-id static_mappo_lidar

# Train MAPPO with an EMA-smoothed actor for more stable evaluation/inference
uv run train.py --algo mappo --config configs/marl_static.yaml --obs-mode lidar --model-id static_mappo_ema --ema-decay 0.995

# Train MADDPG with vision on moving environment
uv run train.py --algo maddpg --config configs/marl_moving.yaml --obs-mode vision --model-id moving_maddpg_vision

# Train Double DQN with bilinear fusion, custom steps
uv run train.py --algo ddqn --config configs/marl_mixed.yaml --obs-mode bilinear --model-id mixed_ddqn_bilinear 200000

# Train Multimodal DRL (always uses bilinear internally)
uv run train.py --algo multimodal --config configs/marl_static.yaml --model-id static_multimodal
```

## Inference

```bash
# Live simulation (opens Arcade window)
uv run inference.py models/static_mappo_lidar configs/marl_static.yaml

# Force EMA actor weights at inference time
uv run inference.py models/static_mappo_ema configs/marl_static.yaml --policy ema

# Video recording
uv run inference.py models/static_mappo_lidar configs/marl_static.yaml --mode video
```

## New Learning Feature: EMA Policy Averaging

MAPPO in `RL-1` now supports an Exponential Moving Average (EMA) copy of the actor. EMA is more common in supervised and self-supervised learning than in small RL repos, so it is a good "borrowed from another domain" technique to experiment with.

- `--ema-decay 0` keeps the original behavior.
- `--ema-decay 0.99` to `0.999` keeps a smoothed actor alongside the online actor during training.
- Checkpoints save both the online actor and the EMA actor.
- Inference can use `--policy ema`, `--policy online`, or `--policy auto`.

## TensorBoard

```bash
tensorboard --logdir logs/
```

## Project Structure

```
RL-1/
├── configs/              3 environment configs (static, moving, mixed)
├── nav/                  Environment engine (physics, LIDAR, rendering)
├── algorithms/
│   ├── mappo/            Multi-Agent PPO
│   ├── maddpg/           Multi-Agent DDPG
│   ├── ddqn/             Double Deep Q-Learning
│   └── multimodal/       Multimodal DRL (cross-attention fusion)
├── networks/             Observation encoders (LIDAR, Faster R-CNN, bilinear)
├── train.py              Unified training CLI
├── inference.py          Inference with live viz / video
├── models/               Saved model weights
├── videos/               Training evaluation videos
├── movies/               Inference videos
└── logs/                 TensorBoard logs
```
