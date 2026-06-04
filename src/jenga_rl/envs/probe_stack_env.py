"""Active-probing Panda Jenga environment (T3: full physics, pure push).

This is the project's main environment.  It implements a human-like Jenga
strategy on top of a real PyBullet physics tower:

1.  **Probe** a candidate block -- the Franka Panda *physically* nudges the
    block's exposed end and the environment measures how far it actually moved
    and how much it resisted.
2.  **Extract** a chosen block -- the robot *physically* pushes the block all
    the way out of the tower.  Whether it slides out cleanly or drags its
    neighbours and topples the tower is decided entirely by the simulation.

The looseness of each block is **physical**: every block is given a randomised
lateral friction at reset.  Loose (low-friction) blocks slide out cleanly;
tight (high-friction) blocks grip their neighbours, so pushing them transfers
force into the tower and risks a collapse.  Nothing about success or collapse is
sampled from a random number generator -- the agent has to *probe* to discover
which blocks are safe to remove.

The class name keeps the historical ``...StackEnv`` suffix for continuity, but
the task is pure push-extraction (there is no top-placement / re-stacking step,
which would require grasping a block the Panda gripper cannot physically open
wide enough to hold).

Action encoding (single discrete space, mask-friendly)::

    0 <= action < N       -> probe block ``action``
    N <= action < 2 * N   -> extract block ``action - N``

where ``N = levels * blocks_per_level`` and a block index decodes as
``divmod(index, blocks_per_level) -> (level, slot)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from jenga_rl.envs.panda_jenga_env import JengaPandaEnv

try:  # pragma: no cover
    import pybullet as p
except ImportError:  # pragma: no cover
    p = None  # type: ignore[assignment]


class JengaPandaProbeStackEnv(JengaPandaEnv):
    """Risk-aware, fully-physical Jenga removal with active probing."""

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    MAX_PROBES_PER_BLOCK = 2

    # Reward shaping constants.
    PROBE_COST = -0.02
    PROBE_SIGNAL_BONUS = 0.18
    EXTRACT_SUCCESS = 1.0
    EXTRACT_STUCK = -0.30
    PROBED_EXTRACTION_BONUS = 0.80
    BLIND_EXTRACTION_PENALTY = -0.80

    # A probe that moves a block more than this (metres, along the push axis) is
    # treated as a "loose" tactile signal; an extraction needs the block to clear
    # this far to count as removed.
    LOOSE_DISPLACEMENT = 0.03
    EXTRACT_THRESHOLD = 0.11

    def __init__(self, *args: Any, fast_dynamics: bool = False, primitive_step_scale: float = 1.0, **kwargs: Any) -> None:
        self.fast_dynamics = fast_dynamics
        self.primitive_step_scale = float(max(0.02, primitive_step_scale))
        self._source_action_count = 0
        self._block_friction = np.zeros(1, dtype=np.float32)
        self._probe_count = np.zeros(1, dtype=np.float32)
        self._probe_displacement = np.zeros(1, dtype=np.float32)
        self._probe_resistance = np.zeros(1, dtype=np.float32)
        self._last_probe_block: tuple[int, int] | None = None
        self._last_action_was_probe = False
        # Physics push parameters.  The force is chosen so that a loose
        # (low-friction) block breaks free while a tight (high-friction) block
        # does not; the speed cap keeps a freed block from launching off the very
        # light (18 g) body under a constant force.
        self._probe_force = 1.4
        self._probe_speed = 0.12
        self._probe_steps = 30
        self._extract_force = 1.9
        self._extract_speed = 0.7
        self._extract_steps = 90
        # When a block is forced out it scrapes the blocks above/below it with a
        # force proportional to its friction.  A tight (high-friction) block drags
        # its neighbours hard enough to topple the tower; a loose one slides clean.
        self._neighbor_drag = 0.3
        super().__init__(*args, **kwargs)
        self._source_action_count = self.config.levels * self.config.blocks_per_level
        self.action_space = spaces.Discrete(2 * self._source_action_count)
        self._reset_probe_arrays()
        # base obs (13) + per-block [count, displacement, resistance] (3N) + phase (2)
        obs_len = 13 + self._source_action_count * 3 + 2
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_len,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _scaled_steps(self, steps: int) -> int:
        return max(1, int(round(steps * self.primitive_step_scale)))

    def _push_direction(self, level: int) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if level % 2 == 0 else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def _load_factor(self, level: int) -> float:
        """Fraction of the tower resting on a block (1.0 at the base, ~0 at the top).

        The friction force a sliding block exerts on its neighbours scales with the
        load it carries, so this is what makes the probe response (displacement
        under a gentle push) a *consistent* predictor of extraction risk: a block
        deep under load both resists the probe and drags its neighbours hard.
        """
        return float((self.config.levels - 1 - level) / max(1, self.config.levels - 1))

    # --------------------------------------------------------------------- API
    def _load_robot(self) -> None:
        """Skip the expensive Panda URDF/IK setup in training-only fast mode.

        Fast mode still uses PyBullet rigid bodies, contact, friction, and gravity
        for the tower.  It only replaces the visible arm trajectory with a
        velocity-capped push force so RL can collect thousands of transitions in a
        reasonable time.  Full-robot rendering keeps the inherited Panda loader.
        """
        if self.fast_dynamics:
            self._robot_id = None
            self._tool_id = None
            return
        super()._load_robot()

    def _ee_state(self):
        if self.fast_dynamics and self._robot_id is None:
            pos = np.asarray([self.config.tower_x - 0.35, self.config.tower_y - 0.25, 0.35], dtype=np.float32)
            vel = np.zeros(3, dtype=np.float32)
            return pos, vel, 0.0
        return super()._ee_state()

    def reset(self, *args: Any, **kwargs: Any):
        self._reset_probe_arrays()
        obs, info = super().reset(*args, **kwargs)
        self._randomize_block_friction()
        self._last_probe_block = None
        self._last_action_was_probe = False
        return self._observation(), self._info(**{k: v for k, v in info.items() if k not in {"legal_actions", "action_mask"}})

    def step(self, action: int):
        action = int(action)
        if action < self._source_action_count:
            return self._probe_step(action)
        return self._extract_step(action - self._source_action_count)

    def legal_actions(self) -> np.ndarray:
        actions: list[int] = []
        for source_action in range(self._source_action_count):
            level, slot = divmod(source_action, self.config.blocks_per_level)
            if not self._is_legal(level, slot):
                continue
            if self._probe_count[source_action] < self.MAX_PROBES_PER_BLOCK:
                actions.append(source_action)  # probe
            actions.append(self._source_action_count + source_action)  # extract
        return np.asarray(actions, dtype=np.int64)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        legal = self.legal_actions()
        if legal.size:
            mask[legal] = True
        return mask

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO-compatible action mask hook."""
        return self.action_mask()

    # ------------------------------------------------------------------ probing
    def _probe_step(self, source_action: int):
        level, slot = divmod(source_action, self.config.blocks_per_level)
        if not self._is_legal(level, slot) or self._probe_count[source_action] >= self.MAX_PROBES_PER_BLOCK:
            return self._observation(), -1.0, False, False, self._info(invalid_action=True)

        self._target_block = (level, slot)
        block_id = self._block_ids[(level, slot)]
        direction = self._push_direction(level)
        before = np.asarray(self._block_position(block_id), dtype=np.float32)

        self._execute_probe_push(level, slot, direction)

        after = np.asarray(self._block_position(block_id), dtype=np.float32)
        displacement = float(abs(np.dot(after - before, direction)))
        resistance = float(np.clip(1.0 - displacement / self.LOOSE_DISPLACEMENT, 0.0, 1.0))

        self._probe_count[source_action] += 1.0
        self._probe_displacement[source_action] = displacement
        self._probe_resistance[source_action] = resistance
        self._last_probe_block = (level, slot)
        self._last_action_was_probe = True

        self._step_simulation(self._scaled_steps(30))
        self.last_stability = self._stability_score()
        self.last_distance = self._distance_to_target()
        collapsed = self._is_collapsed()

        # Make probing an informative but not free action.  A probe that creates
        # measurable displacement tells the agent "this block is loose", while
        # repeated/low-signal probes still cost time.
        signal = float(np.clip(displacement / self.LOOSE_DISPLACEMENT, 0.0, 1.0))
        reward = self.PROBE_COST + self.PROBE_SIGNAL_BONUS * signal
        if collapsed:
            reward += self.config.collapse_penalty
        return self._observation(), float(reward), bool(collapsed), False, self._info(
            probed=True,
            probe_level=level,
            probe_slot=slot,
            probe_displacement=displacement,
            probe_resistance=resistance,
            collapsed=bool(collapsed),
            block_friction=float(self._block_friction[source_action]),
        )

    # --------------------------------------------------------------- extraction
    def _extract_step(self, source_action: int):
        level, slot = divmod(source_action, self.config.blocks_per_level)
        if not self._is_legal(level, slot):
            return self._observation(), -1.0, False, False, self._info(invalid_action=True)

        self._target_block = (level, slot)
        block_id = self._block_ids[(level, slot)]
        direction = self._push_direction(level)
        nominal = self._nominal_block_position(level, slot)
        before_stability = self._stability_score()
        was_probed = bool(self._probe_count[source_action] > 0)

        self._execute_extract_push(level, slot, direction)
        self._step_simulation(self._scaled_steps(self.config.settle_steps))

        pos = np.asarray(self._block_position(block_id), dtype=np.float32)
        displacement = float(abs(np.dot(pos - nominal, direction)))
        fell = bool(pos[2] < nominal[2] - 0.05)
        extracted = bool(displacement > self.EXTRACT_THRESHOLD or fell)

        if extracted:
            p.removeBody(block_id, physicsClientId=self._client_id)
            self._block_ids.pop((level, slot), None)
            self._removed.add((level, slot))
            self.removed_count += 1
            self._step_simulation(self._scaled_steps(40))

        self.last_stability = self._stability_score()
        self.last_distance = self._distance_to_target()
        collapsed = self._is_collapsed()
        completed = self.removed_count >= self.config.max_removed_blocks
        self._last_action_was_probe = False

        reward = 0.2 * (self.last_stability - before_stability)
        if extracted:
            reward += self.EXTRACT_SUCCESS
        else:
            reward += self.EXTRACT_STUCK
        if was_probed and extracted:
            reward += self.PROBED_EXTRACTION_BONUS
        elif not was_probed:
            reward += self.BLIND_EXTRACTION_PENALTY
        if collapsed:
            reward += self.config.collapse_penalty
            if not was_probed:
                reward += self.BLIND_EXTRACTION_PENALTY
        if completed and not collapsed:
            reward += self.config.completion_bonus

        terminated = bool(collapsed or completed or self.legal_actions().size == 0)
        return self._observation(), float(reward), terminated, False, self._info(
            extracted=bool(extracted),
            collapsed=bool(collapsed),
            completed=bool(completed and not collapsed),
            was_probed=was_probed,
            extraction_displacement=displacement,
            block_friction=float(self._block_friction[source_action]),
            blind_extraction=not was_probed,
        )

    # ---------------------------------------------------------------- mechanics
    def _randomize_block_friction(self) -> None:
        """Give every block a physical lateral friction sampled at reset.

        Low friction -> the block slides out cleanly (loose).  High friction ->
        the block grips its neighbours, so pushing it disturbs the tower (tight,
        load-bearing feel).  This is what the agent must discover by probing.
        """
        friction = self.np_random.uniform(0.18, 1.45, size=self._source_action_count).astype(np.float32)
        self._block_friction = friction
        for source_action in range(self._source_action_count):
            key = divmod(source_action, self.config.blocks_per_level)
            body = self._block_ids.get(key)
            if body is None:
                continue
            p.changeDynamics(
                body,
                -1,
                lateralFriction=float(friction[source_action]),
                spinningFriction=0.02,
                rollingFriction=0.02,
                physicsClientId=self._client_id,
            )

    def _block_position(self, block_id: int) -> tuple[float, float, float]:
        pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self._client_id)
        return pos

    def _exposed_end_contact(self, level: int, slot: int, direction: np.ndarray, margin: float) -> np.ndarray:
        """A point just outside the near end face of the block, at block-center height."""
        nominal = self._nominal_block_position(level, slot)
        end = nominal - direction * (self.config.block_length / 2.0)
        # Push near the centre of the selected block's side face.  The earlier
        # high z-offset made the probe look like it was skimming through upper
        # blocks, which is visually unrealistic for Jenga.
        return end - direction * margin

    def _current_block_face_tip(self, block_id: int, direction: np.ndarray, clearance: float = 0.004) -> np.ndarray:
        """Desired probe-tip point for honest visual contact with a moving block.

        The blue probe's *tip* is kept just outside the selected block's near
        face.  As the target block slides, the robot tracks that face instead of
        moving the probe through the block geometry.
        """
        pos = np.asarray(self._block_position(block_id), dtype=np.float32)
        return pos - direction * (self.config.block_length / 2.0 + clearance)


    def _ee_target_for_probe_tip(self, tip_pos: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Convert desired blue-probe tip contact point to a Panda EE target.

        The visible Jenga probe is longer than the stock gripper.  Keeping the
        wrist behind the tower avoids the unrealistic look where the bulky Panda
        hand overlaps neighbouring blocks while only the selected block receives
        the simulated push force.
        """
        return (
            np.asarray(tip_pos, dtype=np.float32)
            - direction * (self.config.probe_center_offset + self.config.probe_half_length)
            - np.asarray([0.0, 0.0, self.config.probe_z_offset], dtype=np.float32)
        )

    def _execute_probe_push(self, level: int, slot: int, direction: np.ndarray) -> None:
        """Gentle nudge against the block's end face (real contact)."""
        if self.fast_dynamics:
            self._fast_force_push(
                self._block_ids[(level, slot)], direction, self._probe_force, self._probe_speed, self._probe_steps
            )
            return
        pre_tip = self._exposed_end_contact(level, slot, direction, margin=0.12)
        touch_tip = self._exposed_end_contact(level, slot, direction, margin=-0.01)
        pre = self._ee_target_for_probe_tip(pre_tip, direction)
        touch = self._ee_target_for_probe_tip(touch_tip, direction)
        block_id = self._block_ids[(level, slot)]
        self._move_ee(pre.tolist(), steps=self._scaled_steps(50))
        self._move_ee_while_force(
            block_id,
            touch.tolist(),
            direction,
            self._probe_force,
            self._probe_speed,
            self._scaled_steps(40),
        )
        self._move_ee(pre.tolist(), steps=self._scaled_steps(35))

    def _execute_extract_push(self, level: int, slot: int, direction: np.ndarray) -> None:
        """Firm push that drives the block all the way out the far side."""
        if self.fast_dynamics:
            source_action = level * self.config.blocks_per_level + slot
            self._fast_force_push(
                self._block_ids[(level, slot)],
                direction,
                self._extract_force,
                self._extract_speed,
                self._extract_steps,
                neighbor_drag=self._neighbor_drag * float(self._block_friction[source_action]) * self._load_factor(level),
                neighbor_level=level,
            )
            return
        nominal = self._nominal_block_position(level, slot)
        pre_tip = self._exposed_end_contact(level, slot, direction, margin=0.12)
        # Stop shortly after the block clears the tower.  Driving much farther
        # makes the very light PyBullet block launch and topple the tower, which
        # is not the controlled Jenga push we want to demonstrate.
        through_tip = nominal + direction * (self.config.block_length / 2.0 + 0.01) + np.asarray([0.0, 0.0, 0.045], dtype=np.float32)
        pre = self._ee_target_for_probe_tip(pre_tip, direction)
        through = self._ee_target_for_probe_tip(through_tip, direction)
        retreat = pre + np.asarray([0.0, 0.0, 0.14], dtype=np.float32)
        block_id = self._block_ids[(level, slot)]
        self._move_ee(pre.tolist(), steps=self._scaled_steps(50))
        self._move_ee_while_force(
            block_id,
            through.tolist(),
            direction,
            self._extract_force,
            self._extract_speed,
            self._scaled_steps(self.config.push_steps),
        )
        self._move_ee(retreat.tolist(), steps=self._scaled_steps(40))

    def _move_ee_while_force(
        self,
        block_id: int,
        target_pos: list[float],
        direction: np.ndarray,
        force: float,
        speed_cap: float,
        steps: int,
    ) -> None:
        """Move the visible Panda probe while applying a controlled physical push.

        In full-robot presentation mode the probe should never visually pass
        through the block.  Each simulation tick recomputes the Panda IK target
        from the *current target block face*, so the long blue probe stays just
        outside the selected block and follows it as it slides out.  The external
        force remains the training-compatible push model, but the visual contact
        is constrained to look like honest Jenga probing/pushing.
        """
        assert self._robot_id is not None
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        # Full-robot render should look like a careful human-like push, not a
        # battering ram.  Fast training keeps the stronger force path elsewhere.
        vec = (direction * (force * 0.65)).tolist()
        capped_speed = min(speed_cap, 0.36)
        for _ in range(max(1, int(steps))):
            desired_tip = self._current_block_face_tip(block_id, direction)
            desired_ee = self._ee_target_for_probe_tip(desired_tip, direction)
            joint_positions = p.calculateInverseKinematics(
                self._robot_id,
                self.PANDA_EE_LINK,
                desired_ee.tolist(),
                orn,
                maxNumIterations=60,
                residualThreshold=1e-4,
                physicsClientId=self._client_id,
            )
            self._sync_tool()
            for joint, value in zip(self.PANDA_ARM_JOINTS, joint_positions[:7], strict=True):
                p.setJointMotorControl2(
                    self._robot_id,
                    joint,
                    p.POSITION_CONTROL,
                    targetPosition=float(value),
                    force=75,
                    positionGain=0.055,
                    velocityGain=0.9,
                    physicsClientId=self._client_id,
                )
            lin_vel, _ = p.getBaseVelocity(block_id, physicsClientId=self._client_id)
            if float(np.dot(np.asarray(lin_vel), direction)) < capped_speed:
                pos = self._block_position(block_id)
                p.applyExternalForce(block_id, -1, vec, list(pos), p.WORLD_FRAME, physicsClientId=self._client_id)
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    def _fast_force_push(
        self,
        block_id: int,
        direction: np.ndarray,
        force: float,
        speed_cap: float,
        steps: int,
        neighbor_drag: float = 0.0,
        neighbor_level: int | None = None,
    ) -> None:
        """Physics-faithful push used for training: drive the block with a
        velocity-capped constant force and let real friction/contacts decide the
        outcome (no arm motion).

        A loose block breaks static friction and cruises out at ``speed_cap``; a
        tight block (static friction > ``force``) barely moves and instead
        transmits the load into its neighbours -- exactly the tactile difference
        a Jenga player feels.  The speed cap stops a freed 18 g block from being
        launched off the table by a constant force.

        ``neighbor_drag`` applies an additional force to the blocks directly above
        and below ``neighbor_level`` so that a tight block scrapes its neighbours
        as it leaves (the same disturbance the real gripper push produces).
        """
        steps = self._scaled_steps(steps)
        vec = (direction * force).tolist()
        drag_vec = (direction * neighbor_drag).tolist()
        drag_bodies: list[int] = []
        if neighbor_drag > 0.0 and neighbor_level is not None:
            for lvl in (neighbor_level - 1, neighbor_level + 1):
                for s in range(self.config.blocks_per_level):
                    body = self._block_ids.get((lvl, s))
                    if body is not None and body != block_id:
                        drag_bodies.append(body)
        drag_speed_cap = 0.25
        for _ in range(steps):
            lin_vel, _ = p.getBaseVelocity(block_id, physicsClientId=self._client_id)
            if float(np.dot(np.asarray(lin_vel), direction)) < speed_cap:
                pos = self._block_position(block_id)
                p.applyExternalForce(block_id, -1, vec, list(pos), p.WORLD_FRAME, physicsClientId=self._client_id)
            for body in drag_bodies:
                bvel, _ = p.getBaseVelocity(body, physicsClientId=self._client_id)
                if float(np.dot(np.asarray(bvel), direction)) < drag_speed_cap:
                    bpos = self._block_position(body)
                    p.applyExternalForce(body, -1, drag_vec, list(bpos), p.WORLD_FRAME, physicsClientId=self._client_id)
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    # --------------------------------------------------------------- arrays/obs
    def _reset_probe_arrays(self) -> None:
        count = max(1, self.config.levels * self.config.blocks_per_level)
        self._source_action_count = count
        self._block_friction = np.zeros(count, dtype=np.float32)
        self._probe_count = np.zeros(count, dtype=np.float32)
        self._probe_displacement = np.zeros(count, dtype=np.float32)
        self._probe_resistance = np.zeros(count, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        probe_features = np.concatenate(
            [
                np.clip(self._probe_count / self.MAX_PROBES_PER_BLOCK, 0.0, 1.0),
                np.clip(self._probe_displacement / 0.18, 0.0, 1.0),
                self._probe_resistance,
            ]
        ).astype(np.float32)
        phase = np.asarray(
            [
                1.0 if self._last_action_was_probe else 0.0,
                0.0 if self._last_probe_block is None else 1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, probe_features, phase]).astype(np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = super()._info()
        info.update(
            {
                "task": "probe_and_push_extract",
                "physics_mode": "fast_force" if self.fast_dynamics else "full_robot",
                "probe_count_total": int(self._probe_count.sum()),
                "removed_count": self.removed_count,
                "last_probe_block": self._last_probe_block,
                "best_observed_block": self.best_observed_block(),
            }
        )
        info.update(extra)
        return info

    def best_observed_block(self) -> tuple[int, int] | None:
        observed = np.asarray(
            [
                action
                for action in np.flatnonzero(self._probe_count > 0)
                if self._is_legal(*divmod(int(action), self.config.blocks_per_level))
            ],
            dtype=np.int64,
        )
        if observed.size == 0:
            return None
        best_action = int(observed[int(self._probe_displacement[observed].argmax())])
        return divmod(best_action, self.config.blocks_per_level)

    def _ansi_render(self) -> str:
        lines = [
            "JengaPandaProbeStackEnv (probe + push-extract)",
            f"removed={self.removed_count} probes={int(self._probe_count.sum())} stability={self.last_stability:.3f}",
        ]
        for level in range(self.config.levels - 1, -1, -1):
            cells = []
            for slot in range(3):
                action = level * self.config.blocks_per_level + slot
                if (level, slot) in self._removed:
                    cells.append("·")
                elif self._probe_count[action] > 0:
                    cells.append("?")
                else:
                    cells.append("█")
            lines.append(f"{level:02d} {'x' if level % 2 == 0 else 'y'} {''.join(cells)}")
        return "\n".join(lines)
