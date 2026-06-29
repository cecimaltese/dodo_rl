<p align="center">
    <img <img width="1879" height="252" alt="TUM_mirmi" src="https://github.com/user-attachments/assets/b1440892-0ecc-4c05-9875-69f6e070998a" />
</p>

# 🦤 dodo_rl — Teaching a Dodo to Walk (and Backflip)

**Reinforcement learning locomotion for the Dodo bipedal robot, built on NVIDIA IsaacLab.**

> *"The original dodo went extinct because it couldn't run. We're fixing that — and adding a backflip for good measure."*

RL locomotion for the Dodo bipedal robot, built as an external IsaacLab project for year S26 team Maltese Invrea, Ghonim, Hucklenbroich, Pickrell.

---

## What Is This?

An IsaacLab project that trains the 8-DOF Dodo bipedal robot to walk (and eventually do acrobatics) using reinforcement learning in simulation (Isaac Sim), with the goal of sim-to-real transfer to the physical robot.

The robot learns entirely from scratch — no hand-crafted gaits, no motion capture. Just a reward signal that says *"go forward, don't fall"*, and 256 parallel robots stumbling their way to competence.

### The Pipeline

```
URDF → USD (Isaac Sim import) → RL Environment → PPO Training (rsl_rl) → Trained Policy → Real Robot
```

## Robot Specs

| Property | Value |
|---|---|
| Name | Dodo |
| Type | Bipedal |
| Total DOF | 8 revolute + 2 fixed (foot soles) |
| Mass | ~4.7 kg |
| Standing height | ~0.45 m |
| Hip motors | 27 Nm max torque |
| Knee/ankle motors | 9 Nm max torque |

### Joint Layout

```
body (base)
├── hip_right (roll, X)  →  upper_leg_right (pitch, Y)  →  lower_leg_right (pitch, Y)  →  foot_right (pitch, Y)
│                                                                                              └── foot_sole_right (fixed)
└── hip_left  (roll, X)  →  upper_leg_left  (pitch, Y)  →  lower_leg_left  (pitch, Y)  →  foot_left  (pitch, Y)
                                                                                               └── foot_sole_left  (fixed)
```

## Registered Environments

| Task ID | Terrain | Use |
|---|---|---|
| `Dodo-Velocity-Flat-v0` | Flat | Training |
| `Dodo-Velocity-Flat-Play-v0` | Flat | Evaluation / visualization |
| `Dodo-Velocity-Rough-v0` | Rough | Training |
| `Dodo-Velocity-Rough-Play-v0` | Rough | Evaluation / visualization |

### Observation Space (36-dim)

Base linear velocity (3) + base angular velocity (3) + projected gravity (3) + velocity commands (3) + joint positions (8) + joint velocities (8) + previous actions (8).

### Action Space (8-dim)

Joint position targets for all 8 revolute joints.

## Getting Started

### Prerequisites

- NVIDIA GPU (tested on RTX 3070, 8 GB VRAM)
- IsaacLab installed and working (`~/IsaacLab`)
- Conda environment `env_isaaclab` active
- A healthy dose of patience (bipedal RL is *hard*)

### Install

```bash
cd ~/IsaacLab
conda activate env_isaaclab
python -m pip install -e ~/dodo_rl/source/dodo_rl
```

### Train

```bash
cd ~/dodo_rl/scripts/rsl_rl
python train.py --task=Dodo-Velocity-Flat-v0 --num_envs=256 --max_iterations=1500 --headless
```

> 💡 **Pro tip:** `--headless` disables rendering and roughly doubles training speed. Your robot doesn't need to see itself fall 10,000 times.

### Play / Evaluate

```bash
cd ~/IsaacLab
python ~/dodo_rl/scripts/rsl_rl/train.py --task=Dodo-Velocity-Flat-v0 --num_envs=64
```

### Monitor Training (W&B) [TBC]

Training metrics are logged to [Weights & Biases](https://wandb.ai/). Make sure you're logged in:

```bash
wandb login
```

Metrics are tracked automatically during training — check your W&B dashboard for live reward curves, episode lengths, and termination stats.

#### What to look for in the curves

| Metric | Good sign | Bad sign |
|---|---|---|
| Mean reward | Trending upward | Stuck negative |
| Episode length | Increasing | Flat at minimum |
| base_contact termination | Dropping toward 0 | Stuck at 1.0 (always falling) |
| track_lin_vel_xy | Climbing | Near zero |
| feet_air_time | Positive and growing | Zero (no stepping gait) |

## Project Structure

```
dodo_rl/
├── scripts/rsl_rl/
│   ├── train.py              # Training script (extends IsaacLab's rsl_rl trainer)
│   ├── play.py               # Evaluation / visualization
│   └── cli_args.py           # CLI argument helpers
├── source/dodo_rl/
│   ├── setup.py
│   ├── pyproject.toml
│   ├── config/extension.toml
│   └── dodo_rl/
│       ├── __init__.py
│       ├── assets/
│       │   ├── dodo.py       # DODO_CFG — robot articulation config
│       │   └── usd/          # USD model + configuration files
│       └── tasks/
│           └── locomotion/
│               ├── __init__.py          # gym.register() calls
│               ├── rough_env_cfg.py     # Base env (rough terrain)
│               ├── flat_env_cfg.py      # Flat terrain variant
│               └── agents/
│                   └── rsl_rl_ppo_cfg.py  # PPO hyperparameters
└── README.md                 # You are here :)
```

### Architecture in One Sentence

The project defines a **robot asset** (USD + actuator config), wraps it in an **RL environment** (observations, rewards, terminations), and trains it with **PPO** via rsl_rl — all as a standalone package that plugs into IsaacLab without modifying it.

## Adapting for New Behaviors

The reward function defines the behavior. Want a different skill? Create a new env config:

- **Backflip** — reward angular velocity around pitch axis + landing upright + full rotation detection
- **Jumping** — reward base height + both feet airborne simultaneously
- **Running** — increase velocity command range

Inherit from `DodoRoughEnvCfg`, override the rewards, register a new gym ID. Same robot, different skills.

## Actuator Configuration

The actuator model uses physics-based PD gains derived from motor specifications:

```
STIFFNESS = ARMATURE × NATURAL_FREQ²
DAMPING   = 2 × DAMPING_RATIO × ARMATURE × NATURAL_FREQ
```

Where `ARMATURE` is the motor's rotor inertia (from datasheet), and `NATURAL_FREQ` / `DAMPING_RATIO` set the control bandwidth. Each motor type on the Dodo has its own armature, effort limit, and velocity limit — see `assets/dodo.py` for the full configuration.

A `DelayedImplicitActuator` is available for sim-to-real transfer, adding configurable command delay to simulate real-world motor latency.

## Known Issues & Notes

- **Warp CUDA warning** (`cuDeviceGetUuid`): cosmetic, doesn't affect training. Known Isaac Sim 5.0 issue.
- **CPU performance profile**: the lab machine may be in `powersave` mode — ask sysadmin to set `performance` for faster training.
- **Fixed joints merged in USD**: `foot_sole_left/right` are merged into `foot_left/right` during URDF→USD conversion (default Isaac Sim behavior). Contact sensing uses `foot_.*` accordingly.

## Credits & Supervision

Supervised by **Dr. Shafeef Omar** and **Dr. Hoan Quang Le** — TUM MIRMI

Built on:
- [NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab) — simulation framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — RL training library (ETH RSL)
- [DoDo Alive! ROS2 framework](https://github.com/) — hardware control interface
