# Colab Training

This project now includes three 9-agent MARL environments designed for stronger coordination training:

- `configs/marl_switch_gate_9agent.yaml`
  Difficulty: easy-to-medium
  Mechanic: one or more agents must hold switches to open a gate.
- `configs/marl_dynamic_roles_9agent.yaml`
  Difficulty: medium
  Mechanic: agents choose corridors and informal roles online.
- `configs/marl_formation_9agent.yaml`
  Difficulty: hard
  Mechanic: 3-agent subteams are rewarded for synchronized arrival.

## Colab Setup

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/<your-user>/<your-repo>.git
%cd /content/<your-repo>/RL-1
!pip install -q -r requirements-colab.txt
```

Optional:

```bash
!nvidia-smi
```

## Recommended MAPPO Runs

MAPPO uses the stronger temporal LiDAR encoder with `--history-length 4`.

```bash
!python train.py --algo mappo --config configs/marl_switch_gate_9agent.yaml --obs-mode lidar --model-id colab_mappo_switch_gate_2m --history-length 4 --ema-decay 0.995 2000000
!python train.py --algo mappo --config configs/marl_dynamic_roles_9agent.yaml --obs-mode lidar --model-id colab_mappo_dynamic_roles_2m --history-length 4 --ema-decay 0.995 2000000
!python train.py --algo mappo --config configs/marl_formation_9agent.yaml --obs-mode lidar --model-id colab_mappo_formation_2m --history-length 4 --ema-decay 0.995 2500000
```

## Recommended MADDPG Runs

MADDPG is included as a comparison baseline on the same 9-agent tasks.

```bash
!python train.py --algo maddpg --config configs/marl_switch_gate_9agent.yaml --obs-mode lidar --model-id colab_maddpg_switch_gate_2m 2000000
!python train.py --algo maddpg --config configs/marl_dynamic_roles_9agent.yaml --obs-mode lidar --model-id colab_maddpg_dynamic_roles_2m 2000000
!python train.py --algo maddpg --config configs/marl_formation_9agent.yaml --obs-mode lidar --model-id colab_maddpg_formation_2m 2000000
```

## Saving Checkpoints to Drive

```bash
!cp -r models /content/drive/MyDrive/rl_project_artifacts/
!cp -r logs /content/drive/MyDrive/rl_project_artifacts/
```

## Notes

- For faster training in Colab, choose a GPU runtime before running installs.
- MAPPO is the primary recommended algorithm for these coordination-heavy environments.
- The switch-gate and synchronized-arrival tasks are deliberately designed so a plain single-agent formulation is a poor fit.
