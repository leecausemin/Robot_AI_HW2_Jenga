"""Single-target joint-control Jenga push environment.

This environment is intentionally separate from the high-level block-selection
T3 task.  A legal target block is sampled at reset; the RL policy controls the
Panda arm joints directly and must probe/push that one block out while avoiding
neighbour disturbance and tower collapse.

Fast-course-project compromise:
- The episode starts with the probe tip near the selected block face, so learning
  focuses on fine joint motion / push intensity rather than long-horizon reaching.
- Action is 7-DOF joint delta control.  The force applied to the block comes from
  the learned tip motion into the block face, so slow careful pushes are rewarded
  and rough pushes disturb neighbours/collapse.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jenga_rl.envs.panda_jenga_env import JengaPandaEnv, PandaJengaConfig

try:  # pragma: no cover
    import pybullet as p
except ImportError:  # pragma: no cover
    p = None  # type: ignore[assignment]


class SingleBlockJointPushEnv(JengaPandaEnv):
    """Joint-delta control for extracting one preselected Jenga block."""

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}
    # A block is only "completed" when it has first moved out of the tower and
    # then physically dropped to the PyBullet floor.  The previous partial-pull
    # threshold stopped before the falling part of the task was visible.
    EXTRACTION_DISTANCE_RATIO = 0.86
    FLOOR_CENTER_HEIGHT_RATIO = 0.95

    def __init__(
        self,
        config: PandaJengaConfig | None = None,
        render_mode: str | None = None,
        *,
        max_episode_steps: int = 80,
        joint_delta_scale: float = 0.035,
        sim_steps_per_action: int = 6,
        near_contact_reset: bool = True,
    ) -> None:
        base = config or PandaJengaConfig(
            levels=6,
            max_removed_blocks=1,
            settle_steps=60,
            motion_steps=45,
            push_steps=60,
            lateral_collapse_threshold=0.16,
            tilt_collapse_threshold=0.38,
            robot_base_x=-0.12,
            robot_base_y=-0.24,
            end_effector="jenga_probe",
        )
        # Keep this env focused and fast even if caller passes a larger T3 config.
        base = replace(
            base,
            max_removed_blocks=1,
            lateral_collapse_threshold=min(base.lateral_collapse_threshold, 0.18),
            tilt_collapse_threshold=min(base.tilt_collapse_threshold, 0.42),
            end_effector="jenga_probe",
        )
        self.max_episode_steps = int(max_episode_steps)
        self.joint_delta_scale = float(joint_delta_scale)
        self.sim_steps_per_action = int(sim_steps_per_action)
        self.near_contact_reset = bool(near_contact_reset)
        self._step_count = 0
        self._target_action = 0
        self._target_body: int | None = None
        self._target_initial_pos = np.zeros(3, dtype=np.float32)
        self._prev_target_disp = 0.0
        self._prev_tip_pos = np.zeros(3, dtype=np.float32)
        self._initial_block_pos: dict[tuple[int, int], np.ndarray] = {}
        self._block_friction: dict[tuple[int, int], float] = {}
        self._last_contact = False
        self._last_tip_face_dist = 1.0
        self._settled_after_full_removal = False
        self._full_removal_reward_given = False
        super().__init__(config=base, render_mode=render_mode)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # joints q/dq (14), tip (3), face vector (3), push direction (3),
        # target disp (1), target height (1), fully removed flag (1), floor flag (1),
        # neighbor motion (1), stability (1), contact flag (1), target level/slot normalized (2)
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(32,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._settled_after_full_removal = False
        self._full_removal_reward_given = False
        self._randomize_block_friction()
        self._sample_target_block()
        if self.near_contact_reset:
            self._place_probe_near_target()
            self._step_simulation(20)
        # Snapshot after the curriculum placement, so neighbour-motion penalties
        # measure damage caused by the learned policy rather than harmless reset
        # settling noise.
        self._target_initial_pos = np.asarray(self._block_position(self._target_body), dtype=np.float32)
        self._snapshot_initial_blocks()
        self._prev_tip_pos = self._probe_tip_position()
        self._prev_target_disp = self._target_displacement()
        self.last_stability = self._stability_score()
        self.last_distance = self._tip_face_distance()
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self._step_count += 1

        before_disp = self._target_displacement()
        before_neighbor = self._neighbor_motion()
        before_stability = self._stability_score()
        before_tip = self._probe_tip_position()

        self._apply_joint_delta(action)

        after_tip = self._probe_tip_position()
        tip_delta = after_tip - before_tip
        self._apply_contact_push(tip_delta)
        self._step_simulation(2)

        target_disp = self._target_displacement()
        progress = max(0.0, target_disp - before_disp)
        neighbor = self._neighbor_motion()
        neighbor_delta = max(0.0, neighbor - before_neighbor)
        collapsed = self._is_collapsed()
        fully_removed = self._is_target_fully_removed()
        if fully_removed and not self._settled_after_full_removal:
            self._step_simulation(90)
            self._settled_after_full_removal = True
            target_disp = self._target_displacement()
            neighbor = self._neighbor_motion()
            collapsed = self._is_collapsed()
        floor_dropped = self._is_target_on_floor()
        completed = bool(fully_removed and floor_dropped and not collapsed)
        contact = self._last_contact
        dist = self._tip_face_distance()
        self.last_distance = dist
        self.last_stability = self._stability_score()

        push_alignment = float(np.dot(tip_delta, self._target_push_direction()))

        # Dense reward: approach/contact, slow target progress, strong penalty for disturbing neighbours.
        reward = 0.0
        reward += 14.0 * progress
        reward += 0.12 if contact else -0.08 * min(4.0, dist)
        reward += 0.40 * max(0.0, push_alignment)
        reward -= 8.0 * neighbor_delta
        reward -= 0.015 * float(np.square(action).mean())
        reward += 0.3 * (self.last_stability - before_stability)
        if fully_removed and not collapsed and not self._full_removal_reward_given:
            reward += 2.5
            self._full_removal_reward_given = True
        if completed and neighbor < 0.075:
            reward += 14.0
        elif completed:
            reward += 7.0
        if collapsed:
            reward -= 10.0

        terminated = bool(collapsed or completed)
        truncated = self._step_count >= self.max_episode_steps
        info = self._info(
            extracted=bool(fully_removed),
            floor_dropped=bool(floor_dropped),
            collapsed=bool(collapsed),
            completed=bool(completed),
            target_displacement=float(target_disp),
            full_extraction_distance=float(self._full_extraction_distance()),
            target_height=float(self._target_height()),
            floor_center_height=float(self._floor_center_height()),
            neighbor_motion=float(neighbor),
            contact=bool(contact),
            tip_face_distance=float(dist),
            target_friction=float(self._block_friction.get(self._target_block or (0, 0), 0.0)),
        )
        return self._observation(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- mechanics
    def _sample_target_block(self) -> None:
        candidates: list[tuple[int, int]] = []
        # Avoid top layer and bottom layer in the first quick curriculum.
        for level in range(1, max(2, self.config.levels - self.config.locked_top_levels)):
            for slot in range(self.config.blocks_per_level):
                if self._is_legal(level, slot):
                    candidates.append((level, slot))
        idx = int(self.np_random.integers(0, len(candidates)))
        self._target_block = candidates[idx]
        self._target_action = self._target_block[0] * self.config.blocks_per_level + self._target_block[1]
        self._target_body = self._block_ids[self._target_block]
        self._target_initial_pos = np.asarray(self._block_position(self._target_body), dtype=np.float32)
        # Visual marker: selected block is greenish.
        p.changeVisualShape(
            self._target_body,
            -1,
            rgbaColor=[0.35, 0.95, 0.28, 1.0],
            physicsClientId=self._client_id,
        )

    def _snapshot_initial_blocks(self) -> None:
        self._initial_block_pos = {
            key: np.asarray(self._block_position(body), dtype=np.float32)
            for key, body in self._block_ids.items()
        }

    def _randomize_block_friction(self) -> None:
        self._block_friction = {}
        for key, body in self._block_ids.items():
            friction = float(self.np_random.uniform(0.25, 1.45))
            self._block_friction[key] = friction
            p.changeDynamics(
                body,
                -1,
                lateralFriction=friction,
                spinningFriction=0.02,
                rollingFriction=0.02,
                physicsClientId=self._client_id,
            )

    def _place_probe_near_target(self) -> None:
        assert self._target_body is not None
        direction = self._target_push_direction()
        # Start almost at the block face.  This is a fast curriculum: the policy
        # still controls all 7 joints, but it learns the careful extraction push
        # rather than spending most of a short assignment run learning reaching.
        face_tip = self._current_block_face_tip(clearance=0.006)
        ee_target = self._ee_target_for_tip(face_tip, direction)
        # Position-only IK is more reliable for the two possible Jenga push axes:
        # orientation-constrained IK can miss high/side targets by 10+ cm, which
        # made early PPO runs learn "do nothing" because contact never occurred.
        joints = p.calculateInverseKinematics(
            self._robot_id,
            self.PANDA_EE_LINK,
            ee_target.tolist(),
            maxNumIterations=300,
            residualThreshold=1e-5,
            physicsClientId=self._client_id,
        )
        for joint, value in zip(self.PANDA_ARM_JOINTS, joints[:7], strict=True):
            p.resetJointState(self._robot_id, joint, float(value), physicsClientId=self._client_id)
            p.setJointMotorControl2(
                self._robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(value),
                force=65,
                positionGain=0.08,
                velocityGain=0.85,
                physicsClientId=self._client_id,
            )
        self._sync_tool()

    def _apply_joint_delta(self, action: np.ndarray) -> None:
        assert self._robot_id is not None
        for joint, delta in zip(self.PANDA_ARM_JOINTS, action, strict=True):
            current = p.getJointState(self._robot_id, joint, physicsClientId=self._client_id)[0]
            p.setJointMotorControl2(
                self._robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(current + self.joint_delta_scale * delta),
                force=65,
                positionGain=0.07,
                velocityGain=0.85,
                physicsClientId=self._client_id,
            )
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    def _apply_contact_push(self, tip_delta: np.ndarray) -> None:
        assert self._target_body is not None
        direction = self._target_push_direction()
        face = self._current_block_face_tip(clearance=0.004)
        tip = self._probe_tip_position()
        error = face - tip
        axial_gap = float(np.dot(error, direction))
        lateral_error = error - direction * axial_gap
        dist = float(np.linalg.norm(error))
        lateral_dist = float(np.linalg.norm(lateral_error))
        forward = float(np.dot(tip_delta, direction))
        self._last_tip_face_dist = dist
        self._last_contact = bool(lateral_dist < 0.055 and -0.030 < axial_gap < 0.085 and forward > -0.0002)
        if not self._last_contact:
            return
        # Learned joint motion supplies push intensity.  Fast/large movements disturb neighbours.
        speed_force = np.clip(0.45 + 130.0 * max(0.0, forward), 0.35, 2.8)
        vec = (direction * float(speed_force)).tolist()
        # Tight blocks scrape neighbouring layers more, making sensitivity real.
        friction = self._block_friction.get(self._target_block or (0, 0), 1.0)
        drag = 0.18 * friction * self._load_factor(self._target_block[0] if self._target_block else 0) * speed_force
        for _ in range(5):
            pos = self._block_position(self._target_body)
            p.applyExternalForce(self._target_body, -1, vec, list(pos), p.WORLD_FRAME, physicsClientId=self._client_id)
            if drag > 0.02 and self._target_block is not None:
                for level in (self._target_block[0] - 1, self._target_block[0] + 1):
                    for slot in range(self.config.blocks_per_level):
                        body = self._block_ids.get((level, slot))
                        if body is not None and body != self._target_body:
                            bpos = self._block_position(body)
                            p.applyExternalForce(body, -1, (direction * drag).tolist(), list(bpos), p.WORLD_FRAME, physicsClientId=self._client_id)
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    def _target_push_direction(self) -> np.ndarray:
        if self._target_block is None:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        level, _ = self._target_block
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if level % 2 == 0 else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def _load_factor(self, level: int) -> float:
        return float((self.config.levels - 1 - level) / max(1, self.config.levels - 1))

    def _probe_tip_position(self) -> np.ndarray:
        ee_pos, _, _ = self._ee_state()
        direction = self._target_push_direction()
        return (
            ee_pos
            + direction * (self.config.probe_center_offset + self.config.probe_half_length)
            + np.asarray([0.0, 0.0, self.config.probe_z_offset], dtype=np.float32)
        ).astype(np.float32)

    def _current_block_face_tip(self, clearance: float = 0.004) -> np.ndarray:
        assert self._target_body is not None
        pos = np.asarray(self._block_position(self._target_body), dtype=np.float32)
        return pos - self._target_push_direction() * (self.config.block_length / 2.0 + clearance)

    def _ee_target_for_tip(self, tip_pos: np.ndarray, direction: np.ndarray) -> np.ndarray:
        return (
            np.asarray(tip_pos, dtype=np.float32)
            - direction * (self.config.probe_center_offset + self.config.probe_half_length)
            - np.asarray([0.0, 0.0, self.config.probe_z_offset], dtype=np.float32)
        )

    def _target_displacement(self) -> float:
        if self._target_body is None:
            return 0.0
        pos = np.asarray(self._block_position(self._target_body), dtype=np.float32)
        return float(max(0.0, np.dot(pos - self._target_initial_pos, self._target_push_direction())))

    def _neighbor_motion(self) -> float:
        motions = []
        for key, body in self._block_ids.items():
            if key == self._target_block:
                continue
            initial = self._initial_block_pos.get(key)
            if initial is None:
                continue
            pos = np.asarray(self._block_position(body), dtype=np.float32)
            motions.append(float(np.linalg.norm(pos - initial)))
        return max(motions) if motions else 0.0

    def _tip_face_distance(self) -> float:
        if self._target_body is None:
            return 1.0
        return float(np.linalg.norm(self._probe_tip_position() - self._current_block_face_tip(clearance=0.004)))

    def _is_target_fully_removed(self) -> bool:
        return self._target_displacement() >= self._full_extraction_distance()

    def _full_extraction_distance(self) -> float:
        return float(self.EXTRACTION_DISTANCE_RATIO * self.config.block_length)

    def _target_height(self) -> float:
        if self._target_body is None:
            return 0.0
        return float(self._block_position(self._target_body)[2])

    def _floor_center_height(self) -> float:
        return float(self.FLOOR_CENTER_HEIGHT_RATIO * self.config.block_height)

    def _is_target_on_floor(self) -> bool:
        return self._target_height() <= self._floor_center_height()

    def _block_position(self, block_id: int) -> tuple[float, float, float]:
        pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self._client_id)
        return pos

    def _is_collapsed(self) -> bool:
        """Collapse check for the remaining tower, not the block being extracted."""
        for key, body in self._block_ids.items():
            if key in self._removed or key == self._target_block:
                continue
            pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
            nominal = self._nominal_block_position(*key)
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            if np.linalg.norm(np.asarray(pos[:2]) - nominal[:2]) > self.config.lateral_collapse_threshold:
                return True
            if abs(roll) > self.config.tilt_collapse_threshold or abs(pitch) > self.config.tilt_collapse_threshold:
                return True
        return False

    def _stability_score(self) -> float:
        offsets: list[float] = []
        tilts: list[float] = []
        for key, body in self._block_ids.items():
            if key in self._removed or key == self._target_block:
                continue
            pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
            nominal = self._nominal_block_position(*key)
            offsets.append(float(np.linalg.norm(np.asarray(pos[:2]) - nominal[:2])))
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            tilts.append(max(abs(roll), abs(pitch)))
        if not offsets:
            return 0.0
        offset_score = 1.0 - min(1.0, max(offsets) / self.config.lateral_collapse_threshold)
        tilt_score = 1.0 - min(1.0, max(tilts) / self.config.tilt_collapse_threshold)
        return float(max(0.0, 0.5 * offset_score + 0.5 * tilt_score))

    # --------------------------------------------------------------- obs/info
    def _observation(self) -> np.ndarray:
        assert self._robot_id is not None
        q = []
        dq = []
        for joint in self.PANDA_ARM_JOINTS:
            state = p.getJointState(self._robot_id, joint, physicsClientId=self._client_id)
            q.append(float(state[0]))
            dq.append(float(state[1]))
        tip = self._probe_tip_position()
        face = self._current_block_face_tip(clearance=0.004) if self._target_body is not None else tip
        vec = face - tip
        direction = self._target_push_direction()
        level_norm = 0.0 if self._target_block is None else self._target_block[0] / max(1, self.config.levels - 1)
        slot_norm = 0.0 if self._target_block is None else self._target_block[1] / max(1, self.config.blocks_per_level - 1)
        obs = np.asarray(
            [
                *q,
                *dq,
                *tip,
                *vec,
                *direction,
                self._target_displacement(),
                self._target_height(),
                1.0 if self._is_target_fully_removed() else 0.0,
                1.0 if self._is_target_on_floor() else 0.0,
                self._neighbor_motion(),
                self._stability_score(),
                1.0 if self._last_contact else 0.0,
                level_norm,
                slot_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {
            "task": "single_block_joint_push",
            "physics": "pybullet",
            "robot": "franka_panda",
            "control": "7d_joint_delta_position",
            "target_block": self._target_block,
            "target_displacement": self._target_displacement(),
            "full_extraction_distance": self._full_extraction_distance(),
            "target_height": self._target_height(),
            "floor_center_height": self._floor_center_height(),
            "floor_dropped": self._is_target_on_floor(),
            "neighbor_motion": self._neighbor_motion(),
            "stability_score": self._stability_score(),
            "tip_face_distance": self._tip_face_distance(),
            "contact": self._last_contact,
            "step_count": self._step_count,
        }
        info.update(extra)
        return info

    def _ansi_render(self) -> str:
        return (
            "SingleBlockJointPushEnv\n"
            f"target={self._target_block} disp={self._target_displacement():.3f} "
            f"neighbor={self._neighbor_motion():.3f} stability={self._stability_score():.3f}"
        )
