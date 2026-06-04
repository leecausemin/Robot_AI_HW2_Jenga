"""Two-robot cooperative Jenga extraction with a custom wide gripper.

Robot A (pusher) uses a scripted primitive to expose the selected block. Robot B
uses a Jenga-specific wide gripper and an RL-controlled 7D joint residual +
gripper-close action to grasp, extract, and place/drop the block onto the floor.

This is intentionally a fast course-project environment: grasp stabilization is
implemented with a PyBullet fixed constraint once the custom gripper is aligned
and closed, while the visible robot arm still moves through Panda joint control.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from gymnasium import spaces

from jenga_rl.envs.panda_jenga_env import JengaPandaEnv, PandaJengaConfig

try:  # pragma: no cover
    import pybullet as p
except ImportError:  # pragma: no cover
    p = None  # type: ignore[assignment]


class TwoRobotJengaGripperEnv(JengaPandaEnv):
    """Pusher + custom gripper cooperative Jenga extraction task."""

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    def __init__(
        self,
        config: PandaJengaConfig | None = None,
        render_mode: str | None = None,
        *,
        max_episode_steps: int = 90,
        joint_delta_scale: float = 0.030,
        residual_scale: float = 0.25,
        sim_steps_per_action: int = 5,
    ) -> None:
        base = config or PandaJengaConfig(
            levels=6,
            max_removed_blocks=1,
            settle_steps=50,
            motion_steps=40,
            push_steps=50,
            lateral_collapse_threshold=0.18,
            tilt_collapse_threshold=0.42,
            end_effector="none",
        )
        base = replace(
            base,
            max_removed_blocks=1,
            lateral_collapse_threshold=max(base.lateral_collapse_threshold, 0.30),
            tilt_collapse_threshold=max(base.tilt_collapse_threshold, 0.55),
            end_effector="none",
        )
        self.max_episode_steps = int(max_episode_steps)
        self.joint_delta_scale = float(joint_delta_scale)
        self.residual_scale = float(residual_scale)
        self.sim_steps_per_action = int(sim_steps_per_action)
        self._pusher_tool_half_length = 0.060
        self._pusher_tip_clearance = 0.006
        self._pusher_expose_distance = 0.055
        self._gripper_reach = 0.24
        self._gripper_palm_offset = 0.13
        self._gripper_jaw_offset = 0.045
        self._pusher_approach_displacement = 0.0
        self._pusher_approach_neighbor_disturbance = 0.0
        self._pusher_neighbor_disturbance = 0.0
        self._pusher_target_displacement = 0.0

        self._pusher_id: int | None = None
        self._pusher_tool_id: int | None = None
        self._gripper_palm_id: int | None = None
        self._gripper_left_jaw_id: int | None = None
        self._gripper_right_jaw_id: int | None = None
        self._grasp_constraint_id: int | None = None
        self._target_body: int | None = None
        self._target_initial_pos = np.zeros(3, dtype=np.float32)
        self._initial_block_pos: dict[tuple[int, int], np.ndarray] = {}
        self._step_count = 0
        self._gripper_closed = 0.0
        self._last_grasped = False
        self._ever_grasped = False
        self._last_target_floor = False
        self._last_non_target_contacts = 0
        self._block_friction: dict[tuple[int, int], float] = {}
        super().__init__(config=base, render_mode=render_mode)

        # 7 Panda arm residuals + 1 custom gripper close command.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        # q/dq(14), gripper ee(3), target pos(3), target rel(3), pull dir(3),
        # target disp/height(2), grasp/closed/floor(3), neighbor/stability/contacts(3),
        # level/slot(2), plus room for the custom gripper bookkeeping scalars.
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(36,), dtype=np.float32)

    # ---------------------------------------------------------------- reset/step
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._release_grasp()
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._gripper_closed = 0.0
        self._last_grasped = False
        self._ever_grasped = False
        self._last_target_floor = False
        self._last_non_target_contacts = 0
        self._pusher_approach_displacement = 0.0
        self._pusher_approach_neighbor_disturbance = 0.0
        self._pusher_neighbor_disturbance = 0.0
        self._pusher_target_displacement = 0.0
        self._randomize_block_friction()
        self._sample_target_block()
        self._scripted_pusher_expose()
        self._place_gripper_at_exposed_block()
        self._step_simulation(25)
        self._target_initial_pos = np.asarray(self._block_position(self._target_body), dtype=np.float32)
        self._snapshot_initial_blocks()
        self.last_stability = self._stability_score()
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._step_count += 1
        before_disp = self._target_displacement()
        before_neighbor = self._neighbor_motion()
        before_stability = self._stability_score()

        close_cmd = float(action[7])
        self._gripper_closed = float(np.clip(self._gripper_closed + 0.20 * close_cmd, 0.0, 1.0))
        self._apply_gripper_joint_action(action[:7])
        self._try_grasp()
        self._step_simulation(2)

        target_disp = self._target_displacement()
        progress = max(0.0, target_disp - before_disp)
        neighbor = self._neighbor_motion()
        neighbor_delta = max(0.0, neighbor - before_neighbor)
        collapsed = self._is_collapsed()
        grasped = self._is_grasped()
        floor = self._is_target_on_floor()
        completed = bool(self._ever_grasped and floor and target_disp > 0.14 and not collapsed)
        self._last_target_floor = floor
        self._last_non_target_contacts = self._non_target_gripper_contacts()
        self.last_stability = self._stability_score()

        reward = 0.0
        reward += 7.0 * progress
        reward += 0.25 if grasped else -0.025 * min(5.0, self._gripper_target_distance())
        reward += 0.15 * self._gripper_closed if not grasped else 0.0
        reward -= 14.0 * neighbor_delta
        reward -= 0.20 * float(self._last_non_target_contacts)
        reward -= 0.012 * float(np.square(action[:7]).mean())
        if neighbor > 0.08:
            reward -= 2.0 * (neighbor - 0.08)
        reward += 0.25 * (self.last_stability - before_stability)
        if completed:
            reward += 16.0
        if collapsed:
            reward -= 12.0

        terminated = bool(completed or collapsed)
        truncated = self._step_count >= self.max_episode_steps
        info = self._info(
            grasped=grasped,
            floor_dropped=floor,
            extracted=bool(target_disp > 0.16),
            collapsed=collapsed,
            completed=completed,
            target_displacement=float(target_disp),
            target_height=float(self._target_height()),
            neighbor_motion=float(neighbor),
            non_target_gripper_contacts=int(self._last_non_target_contacts),
            gripper_closed=float(self._gripper_closed),
        )
        return self._observation(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- robot loading/tools
    def _load_robot(self) -> None:
        # Robot A: pusher, placed in front of the tower and aligned with the
        # target layer.  The earlier diagonal base pose made the arm sweep down
        # over the tower, which looked like it was knocking the block out before
        # aiming.
        self._pusher_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [self.config.tower_x, -0.72, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        # Robot B: learner gripper, placed symmetrically behind the tower.
        # The gripper is mounted so the slim red jaws reach inward from this
        # outside side; the bulky Panda body should not visually pass through
        # the Jenga stack.
        self._robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [self.config.tower_x, 0.72, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        rest = [0.0, -0.45, 0.0, -2.35, 0.0, 1.95, 0.78]
        for robot in (self._pusher_id, self._robot_id):
            for joint, value in zip(self.PANDA_ARM_JOINTS, rest, strict=True):
                p.resetJointState(robot, joint, value, physicsClientId=self._client_id)
            for joint in self.PANDA_FINGER_JOINTS:
                p.resetJointState(robot, joint, 0.035, physicsClientId=self._client_id)
            # Disable bulky stock Panda collision; task contacts come from the slim
            # pusher/custom gripper primitives so other blocks are not hit by fingers.
            for link in range(-1, p.getNumJoints(robot, physicsClientId=self._client_id)):
                p.setCollisionFilterGroupMask(robot, link, 0, 0, physicsClientId=self._client_id)
        # Robot A is a scripted helper.  Keep the precise blue pusher fully
        # visible, but make the bulky Panda body translucent so presentation GIFs
        # do not look like the robot arm itself is knocking the tower.
        for link in range(-1, p.getNumJoints(self._pusher_id, physicsClientId=self._client_id)):
            p.changeVisualShape(self._pusher_id, link, rgbaColor=[0.92, 0.92, 0.92, 0.35], physicsClientId=self._client_id)
        self._install_pusher_tool()
        self._install_custom_gripper()
        # Park Robot A high and away from the tower before the task-specific
        # target is sampled.  This prevents recorded reset frames from showing
        # the pusher arm descending through/over the Jenga stack.
        self._move_robot_ee(
            self._pusher_id,
            np.asarray([self.config.tower_x, -0.62, 0.48], dtype=np.float32),
            steps=1,
            snap=True,
        )
        self._move_robot_ee(
            self._robot_id,
            np.asarray([self.config.tower_x, 0.62, 0.50], dtype=np.float32),
            steps=1,
            snap=True,
        )
        self._sync_tool()

    def _install_pusher_tool(self) -> None:
        # Robot A should look like a precise Jenga pusher, not a long bar that
        # sweeps through the whole tower.  The blue tip is visual-only: physical
        # contact during the approach caused the block to slide out before Robot
        # A had actually aimed.  The intentional exposure below is applied only
        # after alignment via a capped force on the target block.
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self._pusher_tool_half_length, 0.0045, 0.0065],
            rgbaColor=[0.05, 0.20, 0.95, 1.0],
            physicsClientId=self._client_id,
        )
        self._pusher_tool_id = p.createMultiBody(
            baseMass=0.03,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )

    def _install_custom_gripper(self) -> None:
        palm_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.035, 0.015], rgbaColor=[0.1, 0.1, 0.1, 1], physicsClientId=self._client_id)
        jaw_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.085, 0.007, 0.018], rgbaColor=[0.95, 0.1, 0.1, 1], physicsClientId=self._client_id)
        # The custom gripper is a task primitive: visible jaws + fixed constraint
        # grasp.  We avoid physical jaw collisions so the gripper itself does not
        # accidentally bulldoze neighbouring blocks; proximity/contact penalties
        # are tracked separately.
        jaw_collision = -1
        self._gripper_palm_id = p.createMultiBody(0.02, -1, palm_visual, [0, 0, 0], physicsClientId=self._client_id)
        self._gripper_left_jaw_id = p.createMultiBody(0.02, jaw_collision, jaw_visual, [0, 0, 0], physicsClientId=self._client_id)
        self._gripper_right_jaw_id = p.createMultiBody(0.02, jaw_collision, jaw_visual, [0, 0, 0], physicsClientId=self._client_id)
        for body in (self._gripper_palm_id, self._gripper_left_jaw_id, self._gripper_right_jaw_id):
            p.changeDynamics(body, -1, lateralFriction=1.4, restitution=0.0, physicsClientId=self._client_id)

    def _sync_tool(self) -> None:
        self._sync_pusher_tool()
        self._sync_custom_gripper()

    def _sync_pusher_tool(self) -> None:
        if self._pusher_id is None or self._pusher_tool_id is None:
            return
        state = p.getLinkState(self._pusher_id, self.PANDA_EE_LINK, physicsClientId=self._client_id)
        ee = np.asarray(state[0], dtype=np.float32)
        direction = self._target_push_direction()
        yaw = 0.0 if abs(float(direction[0])) > 0.5 else np.pi / 2
        center = ee + direction * self._pusher_tool_half_length + np.asarray([0.0, 0.0, -0.035], dtype=np.float32)
        p.resetBasePositionAndOrientation(self._pusher_tool_id, center.tolist(), p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=self._client_id)

    def _sync_custom_gripper(self) -> None:
        if self._robot_id is None or self._gripper_palm_id is None:
            return
        ee, _, _ = self._ee_state()
        direction = self._target_push_direction()
        lateral = self._lateral_axis()
        yaw = 0.0 if abs(float(direction[0])) > 0.5 else np.pi / 2
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        # Robot B stands on the +push-direction side.  The Panda wrist stays
        # outside the tower; only the long custom gripper reaches inward.
        # ``contact_center`` is the logical grasp/contact point at the target
        # block.  The palm is rendered farther outside so it does not look like
        # the Panda wrist/body is touching the Jenga stack.
        contact_center = ee - direction * self._gripper_reach + np.asarray([0.0, 0.0, -0.035], dtype=np.float32)
        palm_center = contact_center + direction * self._gripper_palm_offset
        jaw_center = contact_center + direction * self._gripper_jaw_offset
        opening = 0.070 - 0.030 * self._gripper_closed
        p.resetBasePositionAndOrientation(self._gripper_palm_id, palm_center.tolist(), orn, physicsClientId=self._client_id)
        p.resetBasePositionAndOrientation(self._gripper_left_jaw_id, (jaw_center + lateral * opening).tolist(), orn, physicsClientId=self._client_id)
        p.resetBasePositionAndOrientation(self._gripper_right_jaw_id, (jaw_center - lateral * opening).tolist(), orn, physicsClientId=self._client_id)
        if self._grasp_constraint_id is not None and self._target_body is not None:
            p.changeConstraint(self._grasp_constraint_id, contact_center.tolist(), orn, maxForce=120, physicsClientId=self._client_id)

    # ---------------------------------------------------------------- task mechanics
    def _sample_target_block(self) -> None:
        candidates: list[tuple[int, int]] = []
        # Fast curriculum: sample from edge blocks in the lower/middle tower.
        # Center blocks and very high blocks often require full game-state
        # reasoning; this task isolates custom-gripper extraction control.
        for level in (1,):
            for slot in range(self.config.blocks_per_level):
                if slot == 1:
                    continue
                if self._is_legal(level, slot):
                    candidates.append((level, slot))
        self._target_block = candidates[int(self.np_random.integers(0, len(candidates)))]
        self._target_body = self._block_ids[self._target_block]
        p.changeVisualShape(self._target_body, -1, rgbaColor=[0.35, 0.95, 0.28, 1.0], physicsClientId=self._client_id)

    def _randomize_block_friction(self) -> None:
        self._block_friction = {}
        for key, body in self._block_ids.items():
            friction = float(self.np_random.uniform(0.30, 1.20))
            self._block_friction[key] = friction
            p.changeDynamics(body, -1, lateralFriction=friction, spinningFriction=0.02, rollingFriction=0.02, physicsClientId=self._client_id)

    def _scripted_pusher_expose(self) -> None:
        assert self._target_body is not None and self._pusher_id is not None
        direction = self._target_push_direction()
        pre_positions = {key: self._block_position(body) for key, body in self._block_ids.items()}
        face = self._block_position(self._target_body) - direction * (self.config.block_length / 2 + self._pusher_tip_clearance)
        z_offset = np.asarray([0.0, 0.0, -0.035], dtype=np.float32)
        # Keep the visible pusher tip safely outside the tower before the
        # intentional exposure phase.  Since the pusher visual has length, the
        # EE sits one full pusher length behind the desired tip location.
        ee_pre = face - direction * (2.0 * self._pusher_tool_half_length + 0.22) - z_offset
        ee_start = face - direction * (2.0 * self._pusher_tool_half_length) - z_offset
        safe_hover = ee_pre + np.asarray([0.0, 0.0, 0.32], dtype=np.float32)
        safe_lower = ee_pre + np.asarray([0.0, 0.0, 0.12], dtype=np.float32)
        initial_before_approach = self._block_position(self._target_body)

        # Snap to a high safe pose first so the recorded approach does not show
        # the Panda arm swinging down through the tower from its reset pose.
        self._move_robot_ee(self._pusher_id, safe_hover, steps=1, snap=True)
        self._step_simulation(4)
        # Descend while still well outside the tower, then approach horizontally.
        self._move_robot_ee(self._pusher_id, safe_lower, steps=18)
        self._move_robot_ee(self._pusher_id, ee_pre, steps=16)
        self._move_robot_ee(self._pusher_id, ee_start, steps=24)

        self._pusher_approach_displacement = float(
            max(0.0, np.dot(self._block_position(self._target_body) - initial_before_approach, direction))
        )
        approach_disturbances = []
        for key, body in self._block_ids.items():
            if key == self._target_block:
                continue
            approach_disturbances.append(float(np.linalg.norm(self._block_position(body) - pre_positions[key])))
        self._pusher_approach_neighbor_disturbance = max(approach_disturbances) if approach_disturbances else 0.0

        # Controlled exposure.  The visible pusher tip advances along the target
        # block axis, while a small capped force is applied only to the target
        # block.  This keeps Robot A from visually sweeping through the tower.
        initial = self._block_position(self._target_body)
        force = (direction * 1.35).tolist()
        steps = 60
        dofs = list(self.PANDA_ARM_JOINTS)
        for idx in range(steps):
            disp = float(max(0.0, np.dot(self._block_position(self._target_body) - initial, direction)))
            if disp >= self._pusher_expose_distance:
                break
            desired_disp = min(self._pusher_expose_distance, self._pusher_expose_distance * (idx + 1) / steps)
            ee_goal = ee_start + direction * desired_disp
            joints = p.calculateInverseKinematics(
                self._pusher_id,
                self.PANDA_EE_LINK,
                ee_goal.tolist(),
                maxNumIterations=80,
                residualThreshold=1e-4,
                physicsClientId=self._client_id,
            )
            for joint, value in zip(dofs, joints[:7], strict=True):
                p.setJointMotorControl2(
                    self._pusher_id,
                    joint,
                    p.POSITION_CONTROL,
                    targetPosition=float(value),
                    force=80,
                    positionGain=0.08,
                    velocityGain=0.8,
                    physicsClientId=self._client_id,
                )
            lin_vel, _ = p.getBaseVelocity(self._target_body, physicsClientId=self._client_id)
            if float(np.dot(np.asarray(lin_vel, dtype=np.float32), direction)) < 0.16:
                pos = self._block_position(self._target_body) - direction * (self.config.block_length / 2)
                p.applyExternalForce(self._target_body, -1, force, pos.tolist(), p.WORLD_FRAME, physicsClientId=self._client_id)
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()
        self._step_simulation(10)
        self._pusher_target_displacement = float(
            max(0.0, np.dot(self._block_position(self._target_body) - initial, direction))
        )
        disturbances = []
        for key, body in self._block_ids.items():
            if key == self._target_block:
                continue
            disturbances.append(float(np.linalg.norm(self._block_position(body) - pre_positions[key])))
        self._pusher_neighbor_disturbance = max(disturbances) if disturbances else 0.0

    def _place_gripper_at_exposed_block(self) -> None:
        assert self._target_body is not None and self._robot_id is not None
        direction = self._target_push_direction()
        pos = self._block_position(self._target_body)
        z_offset = np.asarray([0.0, 0.0, -0.035], dtype=np.float32)
        # Keep Robot B's body outside the tower: first snap high on the outside
        # side, descend while still clear, then let only the slim gripper jaws
        # enter the target-block envelope.  Because _sync_custom_gripper uses
        # contact_center = ee - direction * self._gripper_reach + z_offset, this
        # EE target places the red gripper contact point on the exposed target
        # block while the Panda wrist remains outside the tower.
        ee_target = pos + direction * self._gripper_reach - z_offset
        ee_pre = ee_target + direction * 0.20
        safe_hover = ee_pre + np.asarray([0.0, 0.0, 0.32], dtype=np.float32)
        safe_lower = ee_pre + np.asarray([0.0, 0.0, 0.13], dtype=np.float32)
        self._move_robot_ee(self._robot_id, safe_hover, steps=1, snap=True)
        self._step_simulation(4)
        self._move_robot_ee(self._robot_id, safe_lower, steps=18)
        self._move_robot_ee(self._robot_id, ee_pre, steps=18)
        self._move_robot_ee(self._robot_id, ee_target, steps=26)
        self._gripper_closed = 0.0
        self._sync_custom_gripper()

    def _apply_gripper_joint_action(self, residual: np.ndarray) -> None:
        assert self._robot_id is not None
        prior = self._jacobian_prior_action()
        command = np.clip(prior + self.residual_scale * residual, -1.0, 1.0)
        for joint, delta in zip(self.PANDA_ARM_JOINTS, command, strict=True):
            current = p.getJointState(self._robot_id, joint, physicsClientId=self._client_id)[0]
            p.setJointMotorControl2(
                self._robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(current + self.joint_delta_scale * delta),
                force=75,
                positionGain=0.07,
                velocityGain=0.85,
                physicsClientId=self._client_id,
            )
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    def _jacobian_prior_action(self) -> np.ndarray:
        # Before grasp: stay aligned. After grasp: pull along the block axis and
        # descend, which makes floor placement learnable within a short run.
        direction = self._target_push_direction().astype(np.float32)
        if not self._is_grasped():
            desired = direction * 0.20
        elif self._target_displacement() < 0.18:
            desired = direction * 0.95
        else:
            desired = direction * 0.45 + np.asarray([0.0, 0.0, -0.85], dtype=np.float32)
        dofs = list(self.PANDA_ARM_JOINTS) + list(self.PANDA_FINGER_JOINTS)
        q = [p.getJointState(self._robot_id, j, physicsClientId=self._client_id)[0] for j in dofs]
        zeros = [0.0] * len(q)
        jac_t, _ = p.calculateJacobian(self._robot_id, self.PANDA_EE_LINK, [0, 0, 0], q, zeros, zeros, physicsClientId=self._client_id)
        j_arm = np.asarray(jac_t, dtype=np.float32)[:, :7]
        action = j_arm.T @ desired
        max_abs = float(np.max(np.abs(action)))
        if max_abs > 1e-6:
            action = action / max_abs
        return action.astype(np.float32)

    def _try_grasp(self) -> None:
        if self._grasp_constraint_id is not None or self._target_body is None:
            return
        if self._gripper_closed < 0.55:
            return
        if self._gripper_target_distance() > 0.13:
            return
        center = self._gripper_center()
        orn = p.getQuaternionFromEuler([0, 0, 0])
        self._grasp_constraint_id = p.createConstraint(
            parentBodyUniqueId=self._target_body,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=center.tolist(),
            childFrameOrientation=orn,
            physicsClientId=self._client_id,
        )
        p.changeConstraint(self._grasp_constraint_id, center.tolist(), maxForce=120, physicsClientId=self._client_id)
        self._last_grasped = True
        self._ever_grasped = True

    def _release_grasp(self) -> None:
        if self._grasp_constraint_id is not None and self._client_id is not None:
            try:
                p.removeConstraint(self._grasp_constraint_id, physicsClientId=self._client_id)
            except Exception:
                pass
        self._grasp_constraint_id = None
        self._last_grasped = False

    def _move_robot_ee(self, robot_id: int, target_pos: np.ndarray, *, steps: int, snap: bool = False) -> None:
        joints = p.calculateInverseKinematics(robot_id, self.PANDA_EE_LINK, target_pos.tolist(), maxNumIterations=180, residualThreshold=1e-4, physicsClientId=self._client_id)
        if snap:
            for joint, value in zip(self.PANDA_ARM_JOINTS, joints[:7], strict=True):
                p.resetJointState(robot_id, joint, float(value), physicsClientId=self._client_id)
                p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=float(value), force=90, positionGain=0.08, velocityGain=0.8, physicsClientId=self._client_id)
            self._sync_tool()
            return
        for _ in range(steps):
            for joint, value in zip(self.PANDA_ARM_JOINTS, joints[:7], strict=True):
                p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=float(value), force=90, positionGain=0.08, velocityGain=0.8, physicsClientId=self._client_id)
            p.stepSimulation(physicsClientId=self._client_id)
            self._sync_tool()
            self._maybe_capture()

    # ---------------------------------------------------------------- measurements
    def _target_push_direction(self) -> np.ndarray:
        if self._target_block is None:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if self._target_block[0] % 2 == 0 else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def _lateral_axis(self) -> np.ndarray:
        direction = self._target_push_direction()
        return np.asarray([-direction[1], direction[0], 0.0], dtype=np.float32)

    def _block_position(self, body: int) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
        return np.asarray(pos, dtype=np.float32)

    def _snapshot_initial_blocks(self) -> None:
        self._initial_block_pos = {key: self._block_position(body) for key, body in self._block_ids.items()}

    def _target_displacement(self) -> float:
        if self._target_body is None:
            return 0.0
        return float(max(0.0, np.dot(self._block_position(self._target_body) - self._target_initial_pos, self._target_push_direction())))

    def _target_height(self) -> float:
        if self._target_body is None:
            return 0.0
        return float(self._block_position(self._target_body)[2])

    def _is_target_on_floor(self) -> bool:
        return self._target_height() <= 0.95 * self.config.block_height

    def _is_grasped(self) -> bool:
        return self._grasp_constraint_id is not None

    def _gripper_center(self) -> np.ndarray:
        ee, _, _ = self._ee_state()
        return ee - self._target_push_direction() * self._gripper_reach + np.asarray([0.0, 0.0, -0.035], dtype=np.float32)

    def _gripper_target_distance(self) -> float:
        if self._target_body is None:
            return 1.0
        return float(np.linalg.norm(self._gripper_center() - self._block_position(self._target_body)))

    def _neighbor_motion(self) -> float:
        motions = []
        for key, body in self._block_ids.items():
            if key == self._target_block:
                continue
            initial = self._initial_block_pos.get(key)
            if initial is None:
                continue
            motions.append(float(np.linalg.norm(self._block_position(body) - initial)))
        return max(motions) if motions else 0.0

    def _non_target_gripper_contacts(self) -> int:
        bodies = [self._gripper_left_jaw_id, self._gripper_right_jaw_id]
        total = 0
        for jaw in bodies:
            if jaw is None:
                continue
            for key, body in self._block_ids.items():
                if key == self._target_block:
                    continue
                total += len(p.getContactPoints(jaw, body, physicsClientId=self._client_id))
        return total

    def _is_collapsed(self) -> bool:
        # For the two-robot extraction task, small neighbour shifts are allowed
        # but penalized via neighbour_motion.  Collapse means a non-target block
        # actually falls/tilts severely, which matches the user's "다른 블록이
        # 떨어지지 않게" requirement better than a tiny lateral-offset threshold.
        for key, body in self._block_ids.items():
            if key == self._target_block:
                continue
            pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=self._client_id)
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            if key[0] > 0 and pos[2] < 0.065:
                return True
            if abs(roll) > 0.90 or abs(pitch) > 0.90:
                return True
        return False

    def _ee_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        assert self._robot_id is not None
        state = p.getLinkState(self._robot_id, self.PANDA_EE_LINK, computeLinkVelocity=True, physicsClientId=self._client_id)
        pos = np.asarray(state[0], dtype=np.float32)
        vel = np.asarray(state[6], dtype=np.float32)
        return pos, vel, self._gripper_closed

    # ---------------------------------------------------------------- obs/info/render
    def _observation(self) -> np.ndarray:
        q, dq = [], []
        for joint in self.PANDA_ARM_JOINTS:
            state = p.getJointState(self._robot_id, joint, physicsClientId=self._client_id)
            q.append(float(state[0]))
            dq.append(float(state[1]))
        ee, _, _ = self._ee_state()
        target = self._block_position(self._target_body) if self._target_body is not None else np.zeros(3, dtype=np.float32)
        direction = self._target_push_direction()
        level_norm = 0.0 if self._target_block is None else self._target_block[0] / max(1, self.config.levels - 1)
        slot_norm = 0.0 if self._target_block is None else self._target_block[1] / max(1, self.config.blocks_per_level - 1)
        obs = np.asarray([
            *q,
            *dq,
            *ee,
            *target,
            *(target - self._gripper_center()),
            *direction,
            self._target_displacement(),
            self._target_height(),
            1.0 if self._is_grasped() else 0.0,
            self._gripper_closed,
            1.0 if self._is_target_on_floor() else 0.0,
            self._neighbor_motion(),
            self._stability_score(),
            float(self._last_non_target_contacts),
            level_norm,
            slot_norm,
        ], dtype=np.float32)
        return obs

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {
            "task": "two_robot_custom_gripper",
            "physics": "pybullet",
            "robots": "pusher_panda + custom_gripper_panda",
            "control": "robot_b_7d_joint_residual_plus_gripper_close",
            "target_block": self._target_block,
            "target_displacement": self._target_displacement(),
            "target_height": self._target_height(),
            "floor_dropped": self._is_target_on_floor(),
            "grasped": self._is_grasped(),
            "ever_grasped": self._ever_grasped,
            "gripper_closed": self._gripper_closed,
            "neighbor_motion": self._neighbor_motion(),
            "pusher_approach_displacement": self._pusher_approach_displacement,
            "pusher_approach_neighbor_disturbance": self._pusher_approach_neighbor_disturbance,
            "pusher_target_displacement": self._pusher_target_displacement,
            "pusher_neighbor_disturbance": self._pusher_neighbor_disturbance,
            "stability_score": self._stability_score(),
            "non_target_gripper_contacts": self._last_non_target_contacts,
            "step_count": self._step_count,
        }
        info.update(extra)
        return info

    def _camera_image(self) -> np.ndarray:
        """Camera angle for the two-robot presentation.

        The base Panda environment looks from the Robot-A/front side, which makes
        the scripted pusher arm appear to descend onto the tower even when it is
        physically outside the stack.  For this cooperative task, look from the
        Robot-B side so the target block, gripper, and pusher tip are visible
        without Robot A occluding the tower.
        """
        width, height = 960, 720
        view = p.computeViewMatrix(
            cameraEyePosition=[1.20, 1.05, 0.78],
            cameraTargetPosition=[0.56, 0.00, 0.27],
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
        return (
            "TwoRobotJengaGripperEnv\n"
            f"target={self._target_block} disp={self._target_displacement():.3f} "
            f"height={self._target_height():.3f} grasp={self._is_grasped()} "
            f"neighbor={self._neighbor_motion():.3f}"
        )
