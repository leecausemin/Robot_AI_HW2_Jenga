# Algorithm plan for the T3 Jenga robot environment

## Main algorithm

The main algorithm is **Maskable PPO**.

```text
Main: Maskable PPO
Baselines: Random, Greedy probe
Optional future comparison: PPO without masking, A2C, DQN
```

## Why Maskable PPO?

Jenga has many invalid actions:

```text
- top layer block
- already removed block
- final support block of a layer
- over-probed block
```

Plain PPO can waste samples on those invalid choices. Maskable PPO keeps the PPO actor-critic structure but samples only legal actions using `env.action_masks()`.

## RL formulation

| RL component | Project implementation |
| --- | --- |
| Environment | `JengaPandaProbeStackEnv` |
| State/Observation | tower stability + removed blocks + probe memory |
| Action | discrete `probe(block)` or `push_extract(block)` |
| Reward | probe signal, extraction reward, blind/stuck/collapse penalties, completion bonus |
| Agent | Maskable PPO MLP policy |
| Constraint | Jenga rule action mask |

## Robot control decomposition

The agent does **not** learn joint torques from scratch.

```text
RL learns: high-level block decision
Robot primitive handles: IK movement + controlled probe/push motion
Physics decides: displacement, extraction, collapse
```

This makes the problem trainable within a course project while still showing a robot arm physically pushing Jenga blocks.

## Final training run

```bash
PYTHONPATH=src .venv/bin/python scripts/train_probe_stack_agents.py \
  --algo maskable_ppo \
  --timesteps 20000 \
  --seed 11 \
  --out runs/t3_maskable_probe20k \
  --levels 10 \
  --max-removed-blocks 3 \
  --primitive-step-scale 0.2
```

Training artifact:

```text
runs/t3_maskable_probe20k/maskable_ppo/model.zip
```

Training summary:

```text
timesteps: 20,000
elapsed: 3,597.8 sec
final rollout ep_rew_mean: about 11.4
best late-stage rollout ep_rew_mean: about 12.2
```

## Evaluation result

20 randomized evaluation episodes:

| Policy | Mean return | Avg survived turns / 3 | Completion rate | Collapse rate | Avg probe count | Avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | -7.834 | 1.90 | 0.25 | 0.05 | 13.65 | 27.00 |
| Greedy probe | -1.071 | 1.85 | 0.25 | 0.05 | 2.60 | 24.35 |
| Maskable PPO 20k | 12.350 | 3.00 | 1.00 | 0.00 | 6.00 | 12.50 |

## Interpretation

The 20k Maskable PPO policy solved the current target setting:

```text
10-level tower
3 legal removals
20/20 completion
0/20 collapse
average 6 probes per episode
```

This is stronger than the previous 5k experiment, where the policy could remove blocks but still collapsed in some randomized episodes.

## If algorithm comparison is required later

If the report needs more algorithm comparison, use this order:

1. Random baseline
2. Greedy probe baseline
3. PPO without masking
4. Maskable PPO

Expected story:

```text
PPO learns the same type of policy but wastes probability mass on invalid Jenga actions.
Maskable PPO is better for this constrained discrete action space.
```
