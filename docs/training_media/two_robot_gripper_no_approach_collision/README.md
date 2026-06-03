# No-Approach-Collision Two-Robot Gripper Media

This media set fixes the earlier artifact where Robot A's blue pusher could physically touch the tower during its approach and accidentally slide the target block out before it had aimed.

Fix:

- Robot A's blue pusher tip is visual-only during approach.
- Intentional exposure happens only after alignment through a capped target-block force.
- Added metrics to separate accidental approach motion from intentional exposure:
  - `pusher_approach_displacement`
  - `pusher_approach_neighbor_disturbance`
  - `pusher_target_displacement`
  - `pusher_neighbor_disturbance`

Success rollout:

```text
GIF: ppo/rollout_seed_237.gif
completed=True
collapsed=False
floor_dropped=True
pusher_approach_displacement=0.0 m
pusher_approach_neighbor_disturbance≈0.00003 m
pusher_target_displacement≈0.058 m
pusher_neighbor_disturbance≈0.00003 m
neighbor_motion≈0.012 m
```

Untrained comparison:

```text
GIF: untrained/rollout_seed_237.gif
completed=False
extracted=False
grasped=False
```
