"""A compact Gymnasium environment for learning Jenga block selection.

The environment is deliberately lightweight: it captures the decision structure of
robotic Jenga (which block to probe/extract next) without requiring a heavy physics
engine.  Stability is estimated from layer support polygons and tower center of mass,
which makes training fast enough for classroom algorithm comparisons.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jenga_rl.envs.config import JengaConfig


class JengaTowerEnv(gym.Env[np.ndarray, int]):
    """Jenga block-removal task with discrete target-block actions.

    Action ``a`` maps to ``(level, slot)`` by ``divmod(a, 3)``.  The robot attempts
    to remove that block.  The episode ends when the tower collapses, the agent
    reaches ``max_removed_blocks``, or no legal actions remain.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        config: JengaConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or JengaConfig()
        self.config.validate()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.config.levels * self.config.blocks_per_level)
        obs_size = self.config.levels * self.config.blocks_per_level + 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.tower = np.ones((self.config.levels, self.config.blocks_per_level), dtype=np.int8)
        self.removed_count = 0
        self.last_margin = 1.0
        self._slot_xy = self._build_slot_coordinates()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.tower.fill(1)
        self.removed_count = 0
        self.last_margin = self._stability_margin(self.tower)
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        level, slot = self._decode_action(action)

        if not self._is_legal(level, slot):
            reward = self.config.invalid_action_penalty
            return self._observation(), reward, False, False, self._info(invalid_action=True)

        candidate = self.tower.copy()
        candidate[level, slot] = 0
        margin_after = self._stability_margin(candidate)
        failure_probability = self._failure_probability(level, slot, margin_after)
        failed = bool(self.np_random.random() < failure_probability)
        collapsed = failed or margin_after <= self.config.collapse_margin

        self.tower = candidate
        self.removed_count += 1
        self.last_margin = max(0.0, margin_after)

        if collapsed:
            reward = self.config.collapse_penalty + 0.15 * self.removed_count
            return self._observation(), float(reward), True, False, self._info(collapsed=True)

        completed = self.removed_count >= self.config.max_removed_blocks or not self.legal_actions().size
        reward = self.config.success_reward + 0.25 * self.last_margin
        if completed:
            reward += self.config.completion_bonus

        return self._observation(), float(reward), completed, False, self._info(success=True)

    def render(self) -> str | None:
        lines = ["JengaTowerEnv", f"removed={self.removed_count} margin={self.last_margin:.3f}"]
        for level in range(self.config.levels - 1, -1, -1):
            blocks = "".join("█" if occupied else "·" for occupied in self.tower[level])
            orientation = "x" if level % 2 else "y"
            lines.append(f"{level:02d} {orientation} {blocks}")
        frame = "\n".join(lines)
        if self.render_mode == "human":
            print(frame)
            return None
        return frame

    def close(self) -> None:
        return None

    def legal_actions(self) -> np.ndarray:
        """Return legal discrete actions for the current tower state."""
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

    def simulate_action_margin(self, action: int) -> float:
        """Estimate post-action stability without mutating the environment."""
        level, slot = self._decode_action(int(action))
        if not self._is_legal(level, slot):
            return -1.0
        candidate = self.tower.copy()
        candidate[level, slot] = 0
        return self._stability_margin(candidate)

    def _decode_action(self, action: int) -> tuple[int, int]:
        if action < 0 or action >= self.action_space.n:
            return -1, -1
        return divmod(action, self.config.blocks_per_level)

    def _is_legal(self, level: int, slot: int) -> bool:
        if level < 0 or level >= self.config.levels or slot < 0 or slot >= self.config.blocks_per_level:
            return False
        if level >= self.config.levels - self.config.locked_top_levels:
            return False
        if self.tower[level, slot] == 0:
            return False
        # Do not allow removing the final block of a support layer.
        return bool(self.tower[level].sum() > 1)

    def _observation(self) -> np.ndarray:
        occupancy = self.tower.astype(np.float32).reshape(-1)
        extras = np.asarray(
            [
                self.removed_count / max(1, self.config.max_removed_blocks),
                np.clip(self.last_margin, 0.0, 1.0),
                self.legal_actions().size / self.action_space.n,
            ],
            dtype=np.float32,
        )
        return np.concatenate([occupancy, extras]).astype(np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "removed_count": self.removed_count,
            "stability_margin": self.last_margin,
            "legal_actions": self.legal_actions(),
            "action_mask": self.action_mask(),
            "config": asdict(self.config),
        }
        info.update(extra)
        return info

    def _build_slot_coordinates(self) -> np.ndarray:
        coords = np.zeros((self.config.levels, self.config.blocks_per_level, 2), dtype=np.float32)
        offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        for level in range(self.config.levels):
            if level % 2 == 0:
                coords[level, :, 1] = offsets
            else:
                coords[level, :, 0] = offsets
        return coords

    def _stability_margin(self, tower: np.ndarray) -> float:
        """Return a conservative normalized support margin for the whole tower."""
        margins: list[float] = []
        for support_level in range(self.config.levels - 1):
            support_blocks = tower[support_level].astype(bool)
            if not support_blocks.any():
                return -1.0

            above_blocks = tower[support_level + 1 :].astype(bool)
            if not above_blocks.any():
                continue

            support_xy = self._slot_xy[support_level, support_blocks]
            above_xy = self._slot_xy[support_level + 1 :][above_blocks]
            com = above_xy.mean(axis=0)

            # The support polygon is approximated as a rectangle around remaining blocks.
            half_width = 0.58
            lower = support_xy.min(axis=0) - half_width
            upper = support_xy.max(axis=0) + half_width
            distances = np.minimum(com - lower, upper - com)
            layer_margin = float(distances.min() / half_width)
            margins.append(layer_margin)

        if not margins:
            return 1.0
        return min(1.0, min(margins))

    def _failure_probability(self, level: int, slot: int, margin_after: float) -> float:
        difficulty = self.config.center_block_penalty if slot == 1 else self.config.side_block_penalty
        depth_factor = 1.0 - level / max(1, self.config.levels - 1)
        instability = max(0.0, self.config.collapse_margin + 0.18 - margin_after)
        probability = difficulty + 0.03 * depth_factor + 2.5 * instability
        probability += self.config.action_failure_noise * float(self.np_random.random())
        return float(np.clip(probability, 0.0, 0.95))
