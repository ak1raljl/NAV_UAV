# NAV_UAV

NAV_UAV is a reinforcement learning experiment project for UAV navigation based on Unreal Engine, AirSim, Gym, and Stable-Baselines3.

The current setup focuses on training a multirotor to move forward, avoid obstacles, and reach a goal point in an AirSim scene.

## Features

- AirSim multirotor environment wrapped as a Gym environment named `airsim_env`
- Continuous-control training with Stable-Baselines3
- Main training entry supports `PPO`, `TD3`, and `SAC`
- Default policy uses `MultiInputPolicy` with `Dict` observations
- Observation includes:
  - `depth`: depth image with shape `(60, 90, 1)`
  - `state`: low-dimensional state vector


## Requirements

- Python 3.8+
- Unreal Engine scene with AirSim enabled
- `torch`
- `gym`
- `stable-baselines3`
- `numpy`
- `opencv-python`
- `PyYAML`
- `pygame`

## Quick Start

1. Launch the Unreal scene and make sure AirSim is running.
2. Edit `cfg/nav_cfg.yaml` if needed.
3. Start training from the project root:

```bash
python train.py
```

## Config

Training config is stored in `cfg/nav_cfg.yaml`:

```yaml
algo: PPO
total_timesteps: 100000
tb_log_name: PPO_0422
env_name: airsim_env
```

- `algo`: training algorithm, one of `PPO`, `TD3`, `SAC`
- `total_timesteps`: total training steps
- `tb_log_name`: TensorBoard run name
- `env_name`: Gym environment name

## Main Files

- `train.py`: main training entry
- `env/airsim_env.py`: AirSim environment implementation
- `scripts/kb_ctrl/kb_ctrl.py`: keyboard control script for manual testing
- `scripts/algo/PPO/`: older custom PPO experiment code

## Notes

- The environment uses continuous actions for forward speed, vertical speed, and yaw rate.
- The reward combines progress, pose regularization, obstacle distance, and safety terms.
- Training outputs are generated under `logs/` and `log_path/`.
