"""PyBullet-backed 3D Jenga environment.

This environment is the presentation-grade version of the project: it still follows
Gymnasium's custom environment API from the lectures, but the tower is built from
rigid bodies in a 3D physics simulator.  Actions select a block to push/extract.
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

try:  # pragma: no cover - import availability is environment-dependent
    import pybullet as p
except ImportError as exc:  # pragma: no cover
    p = None  # type: ignore[assignment]
    _PYBULLET_IMPORT_ERROR = exc
else:
    _PYBULLET_IMPORT_ERROR = None


@dataclass(frozen=True)
class BulletJengaConfig(JengaConfig):
    """Configuration for the 3D PyBullet Jenga environment."""

    levels: int = 12
    max_removed_blocks: int = 8
    time_step: float = 1.0 / 240.0
    settle_steps: int = 120
    push_steps: int = 100
    push_force: float = 18.0
    extraction_distance: float = 1.15
    block_length: float = 0.30
    block_width: float = 0.09
    block_height: float = 0.055
    block_gap: float = 0.004
    lateral_collapse_threshold: float = 0.24
    tilt_collapse_threshold: float = 0.55


class JengaBulletEnv(gym.Env[np.ndarray, int]):
    """3D rigid-body Jenga block extraction task.

    The action is still discrete: ``level * 3 + slot``.  The simulator pushes the
    selected block along the safe extraction axis for that level, removes the block
    if it has moved far enough, then lets the tower settle before computing reward.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    def __init__(
        self,
        config: BulletJengaConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        if p is None:  # pragma: no cover
            raise RuntimeError("JengaBulletEnv requires pybullet. Install with `pip install pybullet`.") from _PYBULLET_IMPORT_ERROR
        super().__init__()
        self.config = config or BulletJengaConfig()
        self.config.validate()
        self.render_mode = render_mode
        self._client_id: int | None = None
        self._block_ids: dict[tuple[int, int], int] = {}
        self._removed: set[tuple[int, int]] = set()
        self.removed_count = 0
        self.last_stability = 1.0
        self._wood_texture_id: int | None = None

        self.action_space = spaces.Discrete(self.config.levels * self.config.blocks_per_level)
        obs_size = self.config.levels * self.config.blocks_per_level + 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self._connect()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_world()
        self._build_tower()
        self._step_simulation(self.config.settle_steps)
        self.removed_count = 0
        self._removed.clear()
        self.last_stability = self._stability_score()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        level, slot = self._decode_action(int(action))
        if not self._is_legal(level, slot):
            return self._observation(), -1.0, False, False, self._info(invalid_action=True)

        block_id = self._block_ids[(level, slot)]
        before_stability = self._stability_score()
        self._push_block(block_id, level)
        extracted = self._is_extracted(block_id, level)

        if extracted:
            p.removeBody(block_id, physicsClientId=self._client_id)
            self._removed.add((level, slot))
            self.removed_count += 1
        else:
            # Failed extraction is not always collapse, but it is bad manipulation.
            self._removed.add((level, slot))

        self._step_simulation(self.config.settle_steps)
        self.last_stability = self._stability_score()
        collapsed = self._is_collapsed()
        completed = self.removed_count >= self.config.max_removed_blocks
        no_actions_left = self.legal_actions().size == 0

        if collapsed:
            reward = self.config.collapse_penalty + 0.2 * self.removed_count
            terminated = True
        elif not extracted:
            reward = -0.5 + 0.2 * self.last_stability
            terminated = False
        else:
            reward = self.config.success_reward + 0.5 * (self.last_stability - before_stability)
            terminated = completed or no_actions_left
            if completed:
                reward += self.config.completion_bonus

        return self._observation(), float(reward), terminated, False, self._info(
            extracted=extracted,
            collapsed=collapsed,
            completed=completed,
        )

    def render(self) -> np.ndarray | str | None:
        if self.render_mode == "ansi":
            return self._ansi_render()
        if self.render_mode == "rgb_array":
            return self._camera_image()
        if self.render_mode == "human":
            # PyBullet GUI draws continuously; return None like Gymnasium human render.
            return None
        return self._camera_image()

    def close(self) -> None:
        if self._client_id is not None:
            try:
                p.disconnect(physicsClientId=self._client_id)
            except Exception:
                pass
            self._client_id = None

    def legal_actions(self) -> np.ndarray:
        actions: list[int] = []
        for level in range(self.config.levels):
            for slot in range(self.config.blocks_per_level):
                if self._is_legal(level, slot):
                    actions.append(level * self.config.blocks_per_level + slot)
        return np.asarray(actions, dtype=np.int64)


    def action_mask(self) -> np.ndarray:
        """Boolean mask where True means the action is legal under Jenga rules."""
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def _connect(self) -> None:
        if self._client_id is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client_id = p.connect(mode)
        p.setTimeStep(self.config.time_step, physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)

    def _reset_world(self) -> None:
        p.resetSimulation(physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self._client_id)
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self._client_id)
        p.createMultiBody(0, plane, physicsClientId=self._client_id)
        self._wood_texture_id = None
        if WOOD_TEXTURE_PATH.exists():
            self._wood_texture_id = p.loadTexture(str(WOOD_TEXTURE_PATH), physicsClientId=self._client_id)
        self._block_ids.clear()

    def _build_tower(self) -> None:
        half_extents = [
            self.config.block_length / 2,
            self.config.block_width / 2,
            self.config.block_height / 2,
        ]
        visual_shape_x = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.32, 0.62, 0.95, 1.0],
            physicsClientId=self._client_id,
        )
        visual_shape_y = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.95, 0.58, 0.10, 1.0],
            physicsClientId=self._client_id,
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._client_id,
        )
        spacing = self.config.block_width + self.config.block_gap
        for level in range(self.config.levels):
            yaw = np.pi / 2 if level % 2 else 0.0
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            z = self.config.block_height / 2 + level * self.config.block_height
            for slot in range(self.config.blocks_per_level):
                offset = (slot - 1) * spacing
                pos = [0.0, offset, z] if level % 2 == 0 else [offset, 0.0, z]
                body_id = p.createMultiBody(
                    baseMass=0.018,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape_x if level % 2 == 0 else visual_shape_y,
                    basePosition=pos,
                    baseOrientation=quat,
                    physicsClientId=self._client_id,
                )
                p.changeDynamics(
                    body_id,
                    -1,
                    lateralFriction=0.72,
                    spinningFriction=0.02,
                    rollingFriction=0.02,
                    restitution=0.02,
                    physicsClientId=self._client_id,
                )
                if self._wood_texture_id is not None:
                    p.changeVisualShape(
                        body_id,
                        -1,
                        textureUniqueId=self._wood_texture_id,
                        rgbaColor=[0.86, 0.55, 0.25, 1.0],
                        specularColor=[0.25, 0.18, 0.12],
                        physicsClientId=self._client_id,
                    )
                self._block_ids[(level, slot)] = body_id

    def _push_block(self, block_id: int, level: int) -> None:
        # Even levels are long along x, so extract along x. Odd levels extract along y.
        direction = np.asarray([1.0, 0.0, 0.0]) if level % 2 == 0 else np.asarray([0.0, 1.0, 0.0])
        # Alternate push side by random sign for small variation.
        sign = 1.0 if self.np_random.random() > 0.5 else -1.0
        force = (direction * sign * self.config.push_force).tolist()
        for _ in range(self.config.push_steps):
            p.applyExternalForce(
                block_id,
                -1,
                forceObj=force,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self._client_id,
            )
            p.stepSimulation(physicsClientId=self._client_id)

    def _step_simulation(self, steps: int) -> None:
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._client_id)

    def _is_extracted(self, block_id: int, level: int) -> bool:
        pos, _ = p.getBasePositionAndOrientation(block_id, physicsClientId=self._client_id)
        axis_value = abs(pos[0]) if level % 2 == 0 else abs(pos[1])
        return axis_value > self.config.extraction_distance * self.config.block_length

    def _is_collapsed(self) -> bool:
        if not self._block_ids:
            return False
        top_z_expected = (self.config.levels - 0.5) * self.config.block_height
        max_z = 0.0
        for key, body_id in self._block_ids.items():
            if key in self._removed:
                continue
            pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client_id)
            max_z = max(max_z, pos[2])
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            if abs(pos[0]) > self.config.lateral_collapse_threshold or abs(pos[1]) > self.config.lateral_collapse_threshold:
                return True
            if abs(roll) > self.config.tilt_collapse_threshold or abs(pitch) > self.config.tilt_collapse_threshold:
                return True
        return max_z < top_z_expected * 0.75

    def _stability_score(self) -> float:
        offsets = []
        tilts = []
        for key, body_id in self._block_ids.items():
            if key in self._removed:
                continue
            pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client_id)
            offsets.append(max(abs(pos[0]), abs(pos[1])))
            roll, pitch, _ = p.getEulerFromQuaternion(quat)
            tilts.append(max(abs(roll), abs(pitch)))
        if not offsets:
            return 0.0
        offset_score = 1.0 - min(1.0, max(offsets) / self.config.lateral_collapse_threshold)
        tilt_score = 1.0 - min(1.0, max(tilts) / self.config.tilt_collapse_threshold)
        return float(max(0.0, 0.5 * offset_score + 0.5 * tilt_score))

    def _decode_action(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.action_space.n:
            return -1, -1
        return divmod(action, self.config.blocks_per_level)

    def _is_legal(self, level: int, slot: int) -> bool:
        if level < 0 or level >= self.config.levels or slot < 0 or slot >= self.config.blocks_per_level:
            return False
        if level >= self.config.levels - self.config.locked_top_levels:
            return False
        if (level, slot) in self._removed:
            return False
        remaining_in_level = sum((level, s) not in self._removed for s in range(self.config.blocks_per_level))
        return remaining_in_level > 1

    def _observation(self) -> np.ndarray:
        occupancy = np.ones((self.config.levels, self.config.blocks_per_level), dtype=np.float32)
        for level, slot in self._removed:
            if 0 <= level < self.config.levels and 0 <= slot < self.config.blocks_per_level:
                occupancy[level, slot] = 0.0
        extras = np.asarray(
            [
                self.removed_count / max(1, self.config.max_removed_blocks),
                self.last_stability,
                self.legal_actions().size / self.action_space.n,
                1.0 if self._is_collapsed() else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([occupancy.reshape(-1), extras]).astype(np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "removed_count": self.removed_count,
            "stability_score": self.last_stability,
            "legal_actions": self.legal_actions(),
            "action_mask": self.action_mask(),
            "physics": "pybullet",
        }
        info.update(extra)
        return info

    def _camera_image(self) -> np.ndarray:
        width, height = 960, 720
        view = p.computeViewMatrix(
            cameraEyePosition=[0.95, -1.15, 0.90],
            cameraTargetPosition=[0.0, 0.0, 0.34],
            cameraUpVector=[0, 0, 1],
        )
        proj = p.computeProjectionMatrixFOV(55, width / height, 0.01, 5.0)
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
        lines = ["JengaBulletEnv", f"removed={self.removed_count} stability={self.last_stability:.3f}"]
        for level in range(self.config.levels - 1, -1, -1):
            blocks = "".join("·" if (level, slot) in self._removed else "█" for slot in range(3))
            lines.append(f"{level:02d} {'x' if level % 2 == 0 else 'y'} {blocks}")
        return "\n".join(lines)
