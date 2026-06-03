"""Configuration values for the Jenga tower environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JengaConfig:
    """Parameters that define one Jenga environment instance.

    The defaults model the standard 18-level, three-block-per-level starting tower.
    Values are intentionally explicit so experiments can report exactly which task
    variant was used.
    """

    levels: int = 18
    blocks_per_level: int = 3
    locked_top_levels: int = 1
    max_removed_blocks: int = 8
    collapse_margin: float = 0.02
    action_failure_noise: float = 0.03
    center_block_penalty: float = 0.16
    side_block_penalty: float = 0.02
    invalid_action_penalty: float = -1.0
    collapse_penalty: float = -8.0
    success_reward: float = 1.0
    completion_bonus: float = 8.0

    def validate(self) -> None:
        if self.levels < 3:
            raise ValueError("levels must be at least 3")
        if self.blocks_per_level != 3:
            raise ValueError("this environment currently assumes three blocks per level")
        if not 0 <= self.locked_top_levels < self.levels:
            raise ValueError("locked_top_levels must be in [0, levels)")
        if self.max_removed_blocks <= 0:
            raise ValueError("max_removed_blocks must be positive")
        if self.action_failure_noise < 0:
            raise ValueError("action_failure_noise must be non-negative")
