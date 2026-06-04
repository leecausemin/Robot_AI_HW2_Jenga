# JengaPandaProbeStackEnv (T3): active probing + physical push extraction

`JengaPandaProbeStackEnv` is the **main environment** for this project.
It is a 3D PyBullet Jenga environment where a Franka Panda robot probes and physically pushes blocks out of a tower.

## Story

A human Jenga player first tests whether a block is loose, then removes the safer one. This environment gives the robot the same strategy:

```text
probe candidate block -> observe displacement/resistance -> push-extract block -> survive without collapse
```

There is no fake top placement in the T3 version. Top placement would require grasping and carrying a block, but the current Panda gripper/block geometry makes that unreliable. So the environment uses an honest **pure push extraction** task.

## Files

```text
src/jenga_rl/envs/probe_stack_env.py   # main T3 environment
src/jenga_rl/envs/panda_jenga_env.py         # Panda + tower + rendering base
scripts/train_probe_stack.py          # PPO / Maskable PPO training
scripts/eval_probe_stack.py       # random / greedy / model evaluation
scripts/record_probe_stack.py        # GIF/frame recording
tests/test_probe_stack_env.py          # T3 regression tests
```

## Environment stack

```text
Robot: Franka Panda; presentation render shifts the base slightly behind the tower
End-effector visual: long slim blue Jenga probe; the wrist stays outside and the probe tip tracks the selected block face instead of passing through it
Physics: PyBullet rigid bodies, friction, gravity, contact/push forces
Interface: Gymnasium
Tower: 10 levels
Goal: remove 3 legal blocks without collapse
Main task: active probing + push extraction
```

## Action space

For `N = levels * blocks_per_level`:

```text
0 <= action < N      : probe block action
N <= action < 2N     : push-extract block action - N
```

The environment exposes legal action information:

```python
info["legal_actions"]
info["action_mask"]
env.action_masks()
```

## Jenga rules in the action mask

```text
1. The top layer is locked.
2. Removed blocks cannot be selected again.
3. The final support block of a layer cannot be removed.
4. Probe actions are limited per block.
```

## Physical looseness

Each block receives randomized friction at reset.

```text
low friction  -> loose, easier to move
high friction -> tight/riskier, more likely to disturb neighbors
```

The agent cannot directly observe friction. It must use probe response:

```text
probe_count
probe_displacement
probe_resistance
best_observed_block
```

## Training vs rendering modes

| Mode | Use | Behavior |
| --- | --- | --- |
| `fast_dynamics=True` | RL training/evaluation | PyBullet tower + velocity-capped push force; robot body skipped for speed |
| `fast_dynamics=False` | presentation GIF/rendering | Panda IK motion + blue probe visual + controlled physical push |

This makes training feasible while still showing a real robot-arm-style demo.

## Final 20k result

```text
model: runs/t3_maskable_probe20k/maskable_ppo/model.zip
timesteps: 20,000
levels: 10
max_removed_blocks: 3
primitive_step_scale: 0.2
```

20-episode evaluation:

| Policy | Mean return | Avg survived / 3 | Completion rate | Collapse rate | Avg probe count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | -7.834 | 1.90 | 0.25 | 0.05 | 13.65 |
| Greedy probe | -1.071 | 1.85 | 0.25 | 0.05 | 2.60 |
| Maskable PPO 20k | 12.350 | 3.00 | 1.00 | 0.00 | 6.00 |

## Presentation media

```text
docs/training_media/t3_full_robot_probe20k/maskable_ppo/rollout.gif
docs/training_media/t3_full_robot_probe20k/untrained/rollout.gif
# learned render seed: 501, untrained collapse seed: 502
docs/training_media/t3_full_robot_probe20k/maskable_ppo_mid.png
docs/training_media/t3_full_robot_probe20k/maskable_ppo_final.png
docs/training_media/t3_full_robot_probe20k/untrained_final.png
```

## Run commands

Train 20k:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_probe_stack.py \
  --algo maskable_ppo \
  --timesteps 20000 \
  --seed 11 \
  --out runs/t3_maskable_probe20k \
  --levels 10 \
  --max-removed-blocks 3 \
  --primitive-step-scale 0.2
```

Evaluate:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_probe_stack.py \
  --episodes 20 \
  --seed 700 \
  --levels 10 \
  --max-removed-blocks 3 \
  --max-steps 30 \
  --runs runs/t3_maskable_probe20k \
  --out docs/training_media/metrics/t3_probe20k_after_training.csv \
  --primitive-step-scale 0.2
```

Record full robot GIF:

```bash
PYTHONPATH=src .venv/bin/python scripts/record_probe_stack.py \
  --policy maskable_ppo \
  --model runs/t3_maskable_probe20k/maskable_ppo/model.zip \
  --seed 701 \
  --out-dir docs/training_media/t3_full_robot_probe20k \
  --levels 10 \
  --max-removed-blocks 3 \
  --max-steps 12 \
  --primitive-step-scale 1.0
```
