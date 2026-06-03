# JengaPandaStackEnv: extract + top placement task

`JengaPandaStackEnv` is the main assignment environment after the task upgrade. It follows the Capstone lecture structure:

```text
Agent -> action -> Environment(World)
Environment = Robot + Task + Physics Engine
Robot = Franka Panda
Physics = PyBullet
Interface = Gymnasium
```

## Task

One episode step represents a full Jenga turn:

1. Select a legal non-top block.
2. Move the Panda end-effector to the block using IK.
3. Extract the block with a planned motion primitive.
4. Carry the block to the tower top.
5. Place it on the top layer.
6. Succeed only if the tower's lower support structure remains stable.

## Action space

```python
action = extraction_action * 3 + placement_slot
extraction_action = source_level * 3 + source_slot
placement_slot = 0, 1, or 2
```

The top source layer is locked by Jenga rules. The environment exposes both:

```python
info["legal_actions"]
info["action_mask"]
```

## Observation space

Observation includes Capstone/panda-gym style robot and object/task state:

- end-effector position
- end-effector velocity
- finger width
- target block position
- tower stability score
- removed ratio
- legal action ratio
- top placement goal position
- holding/placing phase indicator

## Reward

- Approach/reach improvement: small dense shaping
- Extraction success: positive reward
- Top placement success: larger positive reward
- Tower lower-support stability: shaping
- Collapse: large penalty
- Full successful turn: completion bonus

## Why this is better than block-selection only

The previous task only asked the agent to pick a block. This upgraded task requires the environment to model the full Jenga rule loop: extract a legal block and place it back on top. It is closer to the Capstone lecture's robot manipulation tasks such as reach, push, pick-and-place, and stack.

## Demo

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/demo_panda_stack.py --steps 1 --out docs/jenga_panda_stack_after_turn.png
```

Expected demo info:

```text
extracted=True
placed=True
collapsed=False
completed=True
```
