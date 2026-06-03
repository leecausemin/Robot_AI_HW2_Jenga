"""Franka Panda robot arm Jenga environment.

This is the environment aligned with the Capstone lecture: Gymnasium custom env,
PyBullet physics, Panda robot manipulation, robot/object observations, action,
reward, and episode termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from pathlib import Path
from gymnasium import spaces

from jenga_rl.envs.config import JengaConfig

WOOD_TEXTURE_PATH = Path(__file__).resolve().parents[3] / "assets" / "textures" / "synthetic_wood_diff_1k.jpg"

try:  # pragma: no cover
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover
    p = None  # type: ignore[assignment]
    pybullet_data = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class PandaJengaConfig(JengaConfig):
    """Configuration for the Panda robot Jenga task."""

    levels: int = 10
    max_removed_blocks: int = 6
    time_step: float = 1.0 / 240.0
    settle_steps: int = 120
    motion_steps: int = 90
    push_steps: int = 120
    # block_length is matched to 3 * block_width + 2 * block_gap so that a layer of
    # three blocks spans exactly the length of a perpendicular block in the next
    # layer.  This removes the cross-layer overhang the earlier 0.30 value created.
    block_length: float = 0.278
    block_width: float = 0.09
    block_height: float = 0.055
    block_gap: float = 0.004
    tower_x: float = 0.58
    tower_y: float = 0.00
    robot_base_x: float = 0.00
    robot_base_y: float = 0.00
    robot_base_z: float = 0.00
    lateral_collapse_threshold: float = 0.30
    tilt_collapse_threshold: float = 0.60
    extraction_distance: float = 0.24
    distance_threshold: float = 0.035
    end_effector: str = "jenga_probe"
    probe_half_length: float = 0.22
    probe_half_width: float = 0.006
    probe_half_height: float = 0.007
    probe_center_offset: float = 0.21
    probe_z_offset: float = -0.035


class JengaPandaEnv(gym.Env[np.ndarray, int]):
    """Panda robot pushes selected Jenga blocks using IK control.

    The RL action is a high-level decision variable: which block to remove.  This
    follows the lecture's recommendation not to force RL to learn every joint limit,
    collision constraint, and inverse-kinematics detail when the project goal is the
    manipulation task itself.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    PANDA_EE_LINK = 11
    PANDA_ARM_JOINTS = tuple(range(7))
    PANDA_FINGER_JOINTS = (9, 10)

    def __init__(
        self,
        config: PandaJengaConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        if p is None:  # pragma: no cover
            raise RuntimeError("JengaPandaEnv requires pybullet. Install with `pip install pybullet`.") from _IMPORT_ERROR
        super().__init__()
        self.config = config or PandaJengaConfig()
        self.config.validate()
        self.render_mode = render_mode
        self._client_id: int | None = None
        self._robot_id: int | None = None
        self._block_ids: dict[tuple[int, int], int] = {}
        self._removed: set[tuple[int, int]] = set()
        self.removed_count = 0
        self.last_distance = 1.0
        self.last_stability = 1.0
        self._target_block: tuple[int, int] | None = None
        self._wood_texture_id: int | None = None
        self._tool_id: int | None = None
        # Optional in-motion frame capture so demos can show the arm actually
        # moving/pushing rather than a single snapshot per step.
        self._recording = False
        self._record_every = 8
        self._record_counter = 0
        self._frames: list[np.ndarray] = []

        self.action_space = spaces.Discrete(self.config.levels * self.config.blocks_per_level)
        # Robot state: ee pos(3), ee vel(3), finger width(1)
        # Object/task state: target block pos(3), tower stability(1), removed ratio(1), legal ratio(1)
        obs_size = 3 + 3 + 1 + 3 + 1 + 1 + 1
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_size,), dtype=np.float32)
        self._connect()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_world()
        self._load_robot()
        self._build_tower()
        self._removed.clear()
        self.removed_count = 0
        self._target_block = (4, 0) if self.config.levels > 5 else (1, 0)
        self._step_simulation(self.config.settle_steps)
        self.last_stability = self._stability_score()
        self.last_distance = self._distance_to_target()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        level, slot = self._decode_action(int(action))
        if not self._is_legal(level, slot):
            return self._observation(), -1.0, False, False, self._info(invalid_action=True)

        self._target_block = (level, slot)
        before_distance = self._distance_to_target()
        before_stability = self._stability_score()
        self._execute_push(level, slot)
        extracted = self._is_extracted(level, slot)
        if extracted:
            block_id = self._block_ids[(level, slot)]
            p.removeBody(block_id, physicsClientId=self._client_id)
            self._removed.add((level, slot))
            self.removed_count += 1

        self._step_simulation(self.config.settle_steps)
        self.last_distance = self._distance_to_target()
        self.last_stability = self._stability_score()
        collapsed = self._is_collapsed()
        completed = self.removed_count >= self.config.max_removed_blocks

        # Dense reward like panda-gym: improve distance/reach and task success, punish collapse.
        reach_improvement = before_distance - self.last_distance
        reward = 0.5 * reach_improvement + 0.2 * (self.last_stability - before_stability)
        if extracted:
            reward += 1.0
        if collapsed:
            reward += self.config.collapse_penalty
        if completed:
            reward += self.config.completion_bonus

        terminated = collapsed or completed or self.legal_actions().size == 0
        return self._observation(), float(reward), terminated, False, self._info(
            extracted=extracted,
            collapsed=collapsed,
            completed=completed,
            reached=bool(self.last_distance < self.config.distance_threshold),
        )

    def render(self) -> np.ndarray | str | None:
        if self.render_mode == "human":
            return None
        if self.render_mode == "ansi":
            return self._ansi_render()
        self._sync_tool()
        return self._camera_image()

    def close(self) -> None:
        if self._client_id is not None:
            try:
                p.disconnect(physicsClientId=self._client_id)
            except Exception:
                pass
            self._client_id = None

    def enable_recording(self, every: int = 8) -> None:
        """Start capturing camera frames during arm motion/settle for demo GIFs."""
        self._recording = True
        self._record_every = max(1, int(every))
        self._record_counter = 0
        self._frames = []

    def drain_frames(self) -> list[np.ndarray]:
        """Return captured in-motion frames and clear the buffer."""
        frames = self._frames
        self._frames = []
        return frames

    def _maybe_capture(self) -> None:
        if not self._recording:
            return
        if self._record_counter % self._record_every == 0:
            self._frames.append(self._camera_image())
        self._record_counter += 1

    def legal_actions(self) -> np.ndarray:
        actions: list[int] = []
        for level in range(self.config.levels):
            for slot in range(3):
                if self._is_legal(level, slot):
                    actions.append(level * 3 + slot)
        return np.asarray(actions, dtype=np.int64)

    def action_mask(self) -> np.ndarray:
        """Boolean mask where True means the action is legal under Jenga rules."""
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def _connect(self) -> None:
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        # Extra solver iterations give the contact-rich tower margin to stay stable
        # while a block is being pushed against its neighbours.
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._client_id)

    def _reset_world(self) -> None:
        p.resetSimulation(physicsClientId=self._client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self._client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._client_id)
        p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        self._wood_texture_id = None
        if WOOD_TEXTURE_PATH.exists():
            self._wood_texture_id = p.loadTexture(str(WOOD_TEXTURE_PATH), physicsClientId=self._client_id)
        self._robot_id = None
        self._tool_id = None
        self._block_ids.clear()

    def _load_robot(self) -> None:
        start_pos = [self.config.robot_base_x, self.config.robot_base_y, self.config.robot_base_z]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self._robot_id = p.loadURDF("franka_panda/panda.urdf", start_pos, start_orn, useFixedBase=True, physicsClientId=self._client_id)
        rest = [0.0, -0.45, 0.0, -2.35, 0.0, 1.95, 0.78]
        for joint, value in zip(self.PANDA_ARM_JOINTS, rest, strict=True):
            p.resetJointState(self._robot_id, joint, value, physicsClientId=self._client_id)
        for joint in self.PANDA_FINGER_JOINTS:
            p.resetJointState(self._robot_id, joint, 0.035, physicsClientId=self._client_id)
        if self.config.end_effector == "jenga_probe":
            self._install_jenga_probe_tool()
        self._move_ee([0.42, -0.28, 0.42], steps=120)

    def _install_jenga_probe_tool(self) -> None:
        """Attach a slim single-finger tool better suited to Jenga."""
        assert self._robot_id is not None
        # In this project the gripper is modelled as a slim Jenga probe, not as the
        # stock two-finger Panda hand.  Disable collisions for the bulky Panda
        # links so they do not accidentally knock the tower; the active-probing
        # environment applies a controlled physical push while this visual probe
        # shows the robot's motion.
        for link in range(-1, p.getNumJoints(self._robot_id, physicsClientId=self._client_id)):
            p.setCollisionFilterGroupMask(self._robot_id, link, 0, 0, physicsClientId=self._client_id)
        for link in self.PANDA_FINGER_JOINTS:
            try:
                p.changeVisualShape(
                    self._robot_id,
                    link,
                    rgbaColor=[0.25, 0.25, 0.25, 0.06],
                    physicsClientId=self._client_id,
                )
            except Exception:
                pass

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.config.probe_half_length, self.config.probe_half_width, self.config.probe_half_height],
            rgbaColor=[0.05, 0.16, 0.95, 1.0],
            specularColor=[0.6, 0.6, 0.8],
            physicsClientId=self._client_id,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.config.probe_half_length, self.config.probe_half_width, self.config.probe_half_height],
            physicsClientId=self._client_id,
        )
        self._tool_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        p.changeDynamics(self._tool_id, -1, lateralFriction=1.0, restitution=0.0, physicsClientId=self._client_id)
        self._sync_tool()

    def _target_push_direction(self) -> np.ndarray:
        if self._target_block is None:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        level, _ = self._target_block
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if level % 2 == 0 else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def _sync_tool(self) -> None:
        """Keep the Jenga probe visual aligned with the Panda end-effector."""
        if self._tool_id is None or self._robot_id is None:
            return
        ee_pos, _, _ = self._ee_state()
        direction = self._target_push_direction()
        yaw = 0.0 if abs(float(direction[0])) > 0.5 else np.pi / 2
        center = (
            ee_pos
            + direction * self.config.probe_center_offset
            + np.asarray([0.0, 0.0, self.config.probe_z_offset], dtype=np.float32)
        )
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(self._tool_id, center.tolist(), orn, physicsClientId=self._client_id)

    def _build_tower(self) -> None:
        half_extents = [self.config.block_length / 2, self.config.block_width / 2, self.config.block_height / 2]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._client_id)
        visual_x = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.2, 0.55, 0.95, 1], physicsClientId=self._client_id)
        visual_y = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.95, 0.55, 0.08, 1], physicsClientId=self._client_id)
        spacing = self.config.block_width + self.config.block_gap
        for level in range(self.config.levels):
            yaw = np.pi / 2 if level % 2 else 0.0
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            z = self.config.block_height / 2 + level * self.config.block_height
            for slot in range(3):
                offset = (slot - 1) * spacing
                if level % 2 == 0:
                    pos = [self.config.tower_x, self.config.tower_y + offset, z]
                else:
                    pos = [self.config.tower_x + offset, self.config.tower_y, z]
                body = p.createMultiBody(
                    baseMass=0.018,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual_x if level % 2 == 0 else visual_y,
                    basePosition=pos,
                    baseOrientation=quat,
                    physicsClientId=self._client_id,
                )
                p.changeDynamics(body, -1, lateralFriction=0.78, spinningFriction=0.02, rollingFriction=0.02, physicsClientId=self._client_id)
                if self._wood_texture_id is not None:
                    p.changeVisualShape(
                        body,
                        -1,
                        textureUniqueId=self._wood_texture_id,
                        rgbaColor=[0.86, 0.55, 0.25, 1.0],
                        specularColor=[0.25, 0.18, 0.12],
                        physicsClientId=self._client_id,
                    )
                self._block_ids[(level, slot)] = body

    def _execute_push(self, level: int, slot: int) -> None:
        target = self._nominal_block_position(level, slot)
        # Push direction follows the long axis of the selected block.
        direction = np.asarray([1.0, 0.0, 0.0]) if level % 2 == 0 else np.asarray([0.0, 1.0, 0.0])
        # Approach from the side facing the robot to keep the motion visible.
        start = np.asarray(target) - direction * 0.23 + np.asarray([0.0, -0.01, 0.01])
        contact = np.asarray(target) - direction * 0.11 + np.asarray([0.0, -0.01, 0.01])
        finish = np.asarray(target) + direction * 0.17 + np.asarray([0.0, -0.01, 0.01])
        retreat = finish + np.asarray([0.0, -0.10, 0.10])
        for waypoint, steps in [(start, self.config.motion_steps), (contact, 45), (finish, self.config.push_steps), (retreat, 70)]:
            self._move_ee(waypoint.tolist(), steps=steps)

    def _move_ee(self, target_pos: list[float], steps: int) -> None:
        assert self._robot_id is not None
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_positions = p.calculateInverseKinematics(
            self._robot_id,
            self.PANDA_EE_LINK,
            target_pos,
            orn,
            maxNumIterations=80,
            residualThreshold=1e-4,
            physicsClientId=self._client_id,
        )
        for _ in range(steps):
            self._sync_tool()
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
                p.setJointMotorControl2(self._robot_id, joint, p.POSITION_CONTROL, targetPosition=0.01, force=20, physicsClientId=self._client_id)
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    def _step_simulation(self, steps: int) -> None:
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    def _decode_action(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.action_space.n:
            return -1, -1
        return divmod(action, 3)

    def _is_legal(self, level: int, slot: int) -> bool:
        if level < 0 or level >= self.config.levels or slot < 0 or slot >= 3:
            return False
        if level >= self.config.levels - self.config.locked_top_levels:
            return False
        if (level, slot) in self._removed:
            return False
        return sum((level, s) not in self._removed for s in range(3)) > 1

    def _nominal_block_position(self, level: int, slot: int) -> np.ndarray:
        spacing = self.config.block_width + self.config.block_gap
        offset = (slot - 1) * spacing
        z = self.config.block_height / 2 + level * self.config.block_height
        if level % 2 == 0:
            return np.asarray([self.config.tower_x, self.config.tower_y + offset, z], dtype=np.float32)
        return np.asarray([self.config.tower_x + offset, self.config.tower_y, z], dtype=np.float32)

    def _target_block_position(self) -> np.ndarray:
        if self._target_block is None:
            return np.asarray([self.config.tower_x, self.config.tower_y, self.config.block_height], dtype=np.float32)
        body = self._block_ids.get(self._target_block)
        if body is None or self._target_block in self._removed:
            return self._nominal_block_position(*self._target_block)
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
        return np.asarray(pos, dtype=np.float32)

    def _ee_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        assert self._robot_id is not None
        state = p.getLinkState(self._robot_id, self.PANDA_EE_LINK, computeLinkVelocity=True, physicsClientId=self._client_id)
        pos = np.asarray(state[0], dtype=np.float32)
        vel = np.asarray(state[6], dtype=np.float32)
        finger_width = sum(p.getJointState(self._robot_id, j, physicsClientId=self._client_id)[0] for j in self.PANDA_FINGER_JOINTS)
        return pos, vel, float(finger_width)

    def _distance_to_target(self) -> float:
        ee_pos, _, _ = self._ee_state()
        return float(np.linalg.norm(ee_pos - self._target_block_position()))

    def _is_extracted(self, level: int, slot: int) -> bool:
        body = self._block_ids.get((level, slot))
        if body is None:
            return False
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
        nominal = self._nominal_block_position(level, slot)
        direction_idx = 0 if level % 2 == 0 else 1
        return abs(pos[direction_idx] - nominal[direction_idx]) > self.config.extraction_distance * self.config.block_length

    def _is_collapsed(self) -> bool:
        for key, body in self._block_ids.items():
            if key in self._removed:
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
            if key in self._removed:
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

    def _observation(self) -> np.ndarray:
        ee_pos, ee_vel, finger_width = self._ee_state()
        target = self._target_block_position()
        return np.asarray(
            [
                *ee_pos,
                *ee_vel,
                finger_width,
                *target,
                self.last_stability,
                self.removed_count / max(1, self.config.max_removed_blocks),
                self.legal_actions().size / self.action_space.n,
            ],
            dtype=np.float32,
        )

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "physics": "pybullet",
            "robot": "franka_panda",
            "end_effector": self.config.end_effector,
            "control": "ik_target_pose_push",
            "removed_count": self.removed_count,
            "stability_score": self.last_stability,
            "distance_to_target": self.last_distance,
            "legal_actions": self.legal_actions(),
            "action_mask": self.action_mask(),
        }
        info.update(extra)
        return info

    def _camera_image(self) -> np.ndarray:
        width, height = 960, 720
        view = p.computeViewMatrix(
            cameraEyePosition=[1.20, -1.15, 0.78],
            cameraTargetPosition=[0.52, 0.00, 0.28],
            cameraUpVector=[0, 0, 1],
        )
        proj = p.computeProjectionMatrixFOV(58, width / height, 0.01, 5.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._client_id,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def _ansi_render(self) -> str:
        lines = ["JengaPandaEnv", f"robot=Franka Panda removed={self.removed_count} stability={self.last_stability:.3f}"]
        for level in range(self.config.levels - 1, -1, -1):
            blocks = "".join("·" if (level, slot) in self._removed else "█" for slot in range(3))
            lines.append(f"{level:02d} {'x' if level % 2 == 0 else 'y'} {blocks}")
        return "\n".join(lines)
