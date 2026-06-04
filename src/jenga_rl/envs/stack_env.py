"""Panda robot Jenga task with extraction and top placement.

The task models one full Jenga turn: choose a legal non-top block, extract it with
Franka Panda, carry it above the tower, and place it on the new top layer without
collapsing the tower.
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


class JengaPandaStackEnv(JengaPandaEnv):
    """Full-turn Panda Jenga environment: extract a block and place it on top.

    Action encoding::

        action = extraction_action * 3 + placement_slot
        extraction_action = level * 3 + source_slot
        placement_slot in {0, 1, 2}

    This keeps the policy decision high-level enough for a semester project while
    making the task closer to real Jenga rules than extraction-only environments.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    def __init__(self, *args: Any, primitive_step_scale: float = 1.0, **kwargs: Any) -> None:
        self._placement_goal = np.zeros(3, dtype=np.float32)
        self._placed_slots: set[tuple[int, int]] = set()
        self.placed_count = 0
        self.primitive_step_scale = float(max(0.02, primitive_step_scale))
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(self.config.levels * self.config.blocks_per_level * 3)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(17,), dtype=np.float32)

    def _scaled_steps(self, steps: int) -> int:
        """Scale robot primitive duration for short training experiments."""
        return max(1, int(round(steps * self.primitive_step_scale)))

    def reset(self, *args: Any, **kwargs: Any):
        self._placed_slots.clear()
        self.placed_count = 0
        self._placement_goal = self._top_place_position(0)
        obs, info = super().reset(*args, **kwargs)
        return obs, self._info(**{k: v for k, v in info.items() if k not in {"legal_actions", "action_mask"}})

    def step(self, action: int):
        extraction_action, placement_slot = divmod(int(action), 3)
        level, source_slot = divmod(extraction_action, self.config.blocks_per_level)
        placement_level = self._current_placement_level()
        self._placement_goal = self._top_place_position(placement_slot)

        if not self._is_legal(level, source_slot) or (placement_level, placement_slot) in self._placed_slots:
            return self._observation(), -1.0, False, False, self._info(invalid_action=True)

        self._target_block = (level, source_slot)
        before_distance = self._distance_to_target()
        before_stability = self._stability_score()

        extracted = self._extract_block_with_motion_primitive(level, source_slot)
        placed = False
        placement_distance = 1.0

        if extracted:
            block_id = self._block_ids[(level, source_slot)]
            self._removed.add((level, source_slot))
            self.removed_count += 1
            placed = self._carry_block_to_top(block_id, placement_level, placement_slot)
            if placed:
                self._placed_slots.add((placement_level, placement_slot))
                self.placed_count += 1
            placement_distance = self._placement_distance(block_id, placement_level, placement_slot)

        self._step_simulation(self.config.settle_steps)
        self.last_distance = self._distance_to_target()
        self.last_stability = self._stability_score()
        collapsed = self._is_collapsed()
        completed = placed and not collapsed

        reach_improvement = before_distance - self.last_distance
        stability_delta = self.last_stability - before_stability
        reward = 0.3 * reach_improvement + 0.2 * stability_delta
        if extracted:
            reward += 1.0
        else:
            reward -= 0.5
        if placed:
            reward += 2.0 - min(1.0, placement_distance * 8.0)
        if collapsed:
            reward += self.config.collapse_penalty
        if completed:
            reward += self.config.completion_bonus

        terminated = (
            collapsed
            or self.placed_count >= self.config.max_removed_blocks
            or self.removed_count >= self.config.max_removed_blocks
            or self.legal_actions().size == 0
        )
        return self._observation(), float(reward), terminated, False, self._info(
            extracted=bool(extracted),
            placed=bool(placed),
            collapsed=bool(collapsed),
            completed=bool(completed),
            placement_slot=placement_slot,
            placement_level=placement_level,
            placement_distance=float(placement_distance),
        )

    def legal_actions(self) -> np.ndarray:
        actions: list[int] = []
        placement_level = self._current_placement_level()
        for level in range(self.config.levels):
            for source_slot in range(self.config.blocks_per_level):
                if not self._is_legal(level, source_slot):
                    continue
                extraction_action = level * self.config.blocks_per_level + source_slot
                for placement_slot in range(3):
                    if (placement_level, placement_slot) not in self._placed_slots:
                        actions.append(extraction_action * 3 + placement_slot)
        return np.asarray(actions, dtype=np.int64)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def render(self):
        if self.render_mode == "ansi":
            return self._ansi_render()
        return super().render()

    def _current_placement_level(self) -> int:
        return self.config.levels + self.placed_count // self.config.blocks_per_level

    def _top_place_position(self, placement_slot: int) -> np.ndarray:
        level = self._current_placement_level()
        spacing = self.config.block_width + self.config.block_gap
        offset = (placement_slot - 1) * spacing
        z = self.config.block_height / 2 + level * self.config.block_height
        if level % 2 == 0:
            return np.asarray([self.config.tower_x, self.config.tower_y + offset, z], dtype=np.float32)
        return np.asarray([self.config.tower_x + offset, self.config.tower_y, z], dtype=np.float32)

    def _top_place_orientation(self, placement_level: int):
        yaw = np.pi / 2 if placement_level % 2 else 0.0
        return p.getQuaternionFromEuler([0, 0, yaw])


    def _extract_block_with_motion_primitive(self, level: int, source_slot: int) -> bool:
        """Move the selected block out using a planned Panda motion primitive.

        The capstone lecture explicitly contrasts raw joint-level control with
        target-pose + planner/IK control.  This primitive represents that middle
        layer: the policy chooses *which* Jenga move to attempt, while the
        environment executes a feasible extraction motion without making the
        learning problem depend on brittle low-level contact tuning.
        """
        block_id = self._block_ids[(level, source_slot)]
        nominal = self._nominal_block_position(level, source_slot)
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if level % 2 == 0 else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        block_orn = self._top_place_orientation(level)
        approach = nominal - direction * 0.20 + np.asarray([0.0, -0.08, 0.04], dtype=np.float32)
        extract = nominal + direction * 0.36 + np.asarray([0.0, -0.08, 0.05], dtype=np.float32)
        lift = extract + np.asarray([0.0, 0.0, 0.16], dtype=np.float32)

        self._move_ee(approach.tolist(), steps=self._scaled_steps(70))
        self._move_ee_with_block(extract, block_id, np.asarray([0.0, 0.0, -0.055], dtype=np.float32), steps=self._scaled_steps(90), orientation=block_orn)
        self._move_ee_with_block(lift, block_id, np.asarray([0.0, 0.0, -0.055], dtype=np.float32), steps=self._scaled_steps(60), orientation=block_orn)
        p.resetBaseVelocity(block_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client_id)
        return True

    def _carry_block_to_top(self, block_id: int, placement_level: int, placement_slot: int) -> bool:
        place_pos = self._top_place_position(placement_slot)
        place_orn = self._top_place_orientation(placement_level)
        pickup_pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self._client_id)
        pickup = np.asarray(pickup_pos, dtype=np.float32)

        # A lightweight grasp primitive: the robot moves while the block is held at
        # a fixed offset from the end-effector.  This is common in educational
        # manipulation envs when the learning target is task-level decision making.
        lift = pickup + np.asarray([0.0, 0.0, 0.16], dtype=np.float32)
        above_place = place_pos + np.asarray([0.0, 0.0, 0.18], dtype=np.float32)
        lower_place = place_pos + np.asarray([0.0, 0.0, 0.045], dtype=np.float32)
        block_offset = np.asarray([0.0, 0.0, -0.055], dtype=np.float32)

        self._move_ee_with_block(lift, block_id, block_offset, steps=self._scaled_steps(80), orientation=place_orn)
        self._move_ee_with_block(above_place, block_id, block_offset, steps=self._scaled_steps(110), orientation=place_orn)
        self._move_ee_with_block(lower_place, block_id, block_offset, steps=self._scaled_steps(70), orientation=place_orn)

        p.resetBasePositionAndOrientation(
            block_id,
            place_pos.tolist(),
            place_orn,
            physicsClientId=self._client_id,
        )
        p.resetBaseVelocity(block_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client_id)
        self._move_ee((above_place + np.asarray([0.0, -0.12, 0.08], dtype=np.float32)).tolist(), steps=self._scaled_steps(70))
        self._step_simulation(self._scaled_steps(80))
        return self._placement_distance(block_id, placement_level, placement_slot) < 0.06

    def _move_ee_with_block(
        self,
        target_pos: np.ndarray,
        block_id: int,
        block_offset: np.ndarray,
        steps: int,
        orientation,
    ) -> None:
        assert self._robot_id is not None
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_positions = p.calculateInverseKinematics(
            self._robot_id,
            self.PANDA_EE_LINK,
            target_pos.tolist(),
            orn,
            maxNumIterations=80,
            residualThreshold=1e-4,
            physicsClientId=self._client_id,
        )
        for _ in range(steps):
            for joint, value in zip(self.PANDA_ARM_JOINTS, joint_positions[:7], strict=True):
                p.setJointMotorControl2(
                    self._robot_id,
                    joint,
                    p.POSITION_CONTROL,
                    targetPosition=float(value),
                    force=90,
                    positionGain=0.08,
                    velocityGain=0.8,
                    physicsClientId=self._client_id,
                )
            for joint in self.PANDA_FINGER_JOINTS:
                p.setJointMotorControl2(
                    self._robot_id,
                    joint,
                    p.POSITION_CONTROL,
                    targetPosition=0.01,
                    force=20,
                    physicsClientId=self._client_id,
                )
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            ee_pos, _, _ = self._ee_state()
            p.resetBasePositionAndOrientation(
                block_id,
                (ee_pos + block_offset).tolist(),
                orientation,
                physicsClientId=self._client_id,
            )
            p.resetBaseVelocity(block_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client_id)

    def _placement_distance(self, block_id: int, placement_level: int, placement_slot: int) -> float:
        pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self._client_id)
        expected = self._top_place_position(placement_slot)
        return float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - expected))


    def _is_collapsed(self) -> bool:
        """Stack task collapse check focused on lower support failure.

        Top-placement can cause harmless movement in the top two layers.  We only
        terminate as collapse when the lower support structure moves/tilts enough
        that the tower is no longer standing.
        """
        support_cutoff = max(1, self.config.levels - 2)
        for key, body in self._block_ids.items():
            level, _ = key
            if key in self._removed or level >= support_cutoff:
                continue
            pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
            nominal = self._nominal_block_position(*key)
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            if np.linalg.norm(np.asarray(pos[:2]) - nominal[:2]) > 0.42:
                return True
            if abs(roll) > 0.95 or abs(pitch) > 0.95:
                return True
        return False

    def _stability_score(self) -> float:
        support_cutoff = max(1, self.config.levels - 2)
        offsets: list[float] = []
        tilts: list[float] = []
        for key, body in self._block_ids.items():
            level, _ = key
            if key in self._removed or level >= support_cutoff:
                continue
            pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
            nominal = self._nominal_block_position(*key)
            offsets.append(float(np.linalg.norm(np.asarray(pos[:2]) - nominal[:2])))
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            tilts.append(max(abs(roll), abs(pitch)))
        if not offsets:
            return 0.0
        offset_score = 1.0 - min(1.0, max(offsets) / 0.42)
        tilt_score = 1.0 - min(1.0, max(tilts) / 0.95)
        return float(max(0.0, 0.5 * offset_score + 0.5 * tilt_score))

    def _observation(self) -> np.ndarray:
        base = JengaPandaEnv._observation(self)
        holding_or_placing = 1.0 if self._target_block in self._removed else 0.0
        return np.concatenate(
            [
                base,
                self._placement_goal.astype(np.float32),
                np.asarray([holding_or_placing], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = JengaPandaEnv._info(self)
        info.update(
            {
                "task": "extract_and_top_place",
                "placed_count": self.placed_count,
                "placement_goal": self._placement_goal.copy(),
            }
        )
        info.update(extra)
        return info

    def _ansi_render(self) -> str:
        lines = [
            "JengaPandaStackEnv",
            f"robot=Franka Panda removed={self.removed_count} placed={self.placed_count} stability={self.last_stability:.3f}",
        ]
        for level in range(self._current_placement_level(), -1, -1):
            if level >= self.config.levels:
                blocks = "".join("▲" if (level, slot) in self._placed_slots else "·" for slot in range(3))
            else:
                blocks = "".join("·" if (level, slot) in self._removed else "█" for slot in range(3))
            lines.append(f"{level:02d} {'x' if level % 2 == 0 else 'y'} {blocks}")
        return "\n".join(lines)
