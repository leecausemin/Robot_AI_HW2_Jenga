from __future__ import annotations

import numpy as np

from jenga_rl.envs import JengaPandaStackEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def test_stack_env_runs_full_turn_and_renders() -> None:
    env = JengaPandaStackEnv(config=PandaJengaConfig(end_effector="panda_gripper"), render_mode="rgb_array")
    obs, info = env.reset(seed=21)
    assert env.observation_space.contains(obs)
    assert info["task"] == "extract_and_top_place"
    assert info["placed_count"] == 0

    source_level = env.config.levels - 2
    extraction_action = source_level * env.config.blocks_per_level
    action = extraction_action * 3 + 1
    assert bool(info["action_mask"][action])

    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert not truncated
    assert info["extracted"]
    assert info["placed"]
    assert not info["collapsed"]
    assert info["completed"]
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 960, 3)
    env.close()


def test_panda_stack_top_source_blocks_are_illegal() -> None:
    env = JengaPandaStackEnv(config=PandaJengaConfig(end_effector="panda_gripper"))
    _, info = env.reset(seed=22)
    top_level = env.config.levels - 1
    for source_slot in range(env.config.blocks_per_level):
        extraction_action = top_level * env.config.blocks_per_level + source_slot
        for placement_slot in range(3):
            action = extraction_action * 3 + placement_slot
            assert not bool(info["action_mask"][action])
    env.close()
