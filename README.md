# dodo_rl

RL locomotion for the Dodo bipedal robot, built as an external IsaacLab project for year S26 team Maltese Invrea, Ghonim, Hucklenbroich, Pickrell.

## Setup

```bash
cd ~/IsaacLab
conda activate env_isaaclab
python -m pip install -e ~/dodo_rl/source/dodo_rl
```

## Train

```bash
cd ~/IsaacLab
python ~/dodo_rl/scripts/rsl_rl/train.py --task=Dodo-Velocity-Flat-v0 --num_envs=64
```

## Play

```bash
python ~/dodo_rl/scripts/rsl_rl/play.py --task=Dodo-Velocity-Flat-Play-v0
```

## Registered environments

| Task ID | Description |
|---|---|
| Dodo-Velocity-Flat-v0 | Flat terrain, training |
| Dodo-Velocity-Flat-Play-v0 | Flat terrain, evaluation |
| Dodo-Velocity-Rough-v0 | Rough terrain, training |
| Dodo-Velocity-Rough-Play-v0 | Rough terrain, evaluation |
