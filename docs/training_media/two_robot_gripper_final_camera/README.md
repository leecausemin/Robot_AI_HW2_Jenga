# Final two-robot gripper media: safe Robot A path

This version addresses the visual/behavior issue where Robot A appeared to descend into the Jenga tower and push the target block before aiming.

Fixes included:

- Robot A base moved to the front side of the tower instead of a diagonal sweep pose.
- Robot A starts from a high park pose outside the tower.
- Approach path is `safe hover -> outside lower -> horizontal approach`, not a descent through the tower.
- Robot A pusher is visual-only during approach; intentional target exposure is applied only after alignment.
- Two-robot camera is moved to the Robot B side so Robot A does not occlude the tower.

Success rollout:

```text
GIF: ppo/rollout_seed_237.gif
completed=True
collapsed=False
floor_dropped=True
pusher_approach_displacement=0.0 m
pusher_approach_neighbor_disturbance≈0.00002 m
pusher_target_displacement≈0.057 m
pusher_neighbor_disturbance≈0.00003 m
neighbor_motion≈0.018 m
```
