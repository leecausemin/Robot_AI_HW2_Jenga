# Precise Pusher + Custom Gripper Media

This media set uses the refined two-robot environment where Robot A uses a short aligned pusher tip instead of a long bar through the tower, and Robot B is penalized more strongly for disturbing non-target blocks.

- Trained PPO success GIF: `ppo/rollout_seed_237.gif`
- Untrained random failure GIF: `untrained/rollout_seed_237.gif`
- Slide stills: `stills/`

Success rollout final metrics:

```text
completed=True
collapsed=False
floor_dropped=True
target_displacement=0.232 m
target_height=0.019 m
neighbor_motion=0.013 m
pusher_target_displacement=0.049 m
pusher_neighbor_disturbance=0.003 m
non_target_gripper_contacts=0
```
