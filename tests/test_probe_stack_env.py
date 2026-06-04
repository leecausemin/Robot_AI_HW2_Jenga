from __future__ import annotations

import numpy as np

from jenga_rl.envs import JengaPandaProbeStackEnv


def test_probe_stack_env_probe_then_push_extract() -> None:
    env = JengaPandaProbeStackEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=31)
    assert env.observation_space.contains(obs)
    assert info["task"] == "probe_and_push_extract"
    assert info["physics_mode"] == "full_robot"
    assert info["probe_count_total"] == 0

    source_level = env.config.levels - 2
    source_action = source_level * env.config.blocks_per_level
    assert bool(info["action_mask"][source_action])

    obs, reward, terminated, truncated, info = env.step(source_action)
    assert env.observation_space.contains(obs)
    assert info["probed"]
    assert info["probe_count_total"] == 1
    assert not terminated
    assert not truncated

    action = env._source_action_count + source_action
    assert bool(info["action_mask"][action])
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert info["was_probed"]
    assert info["extracted"]
    assert "placed" not in info
    assert isinstance(info["collapsed"], bool)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 960, 3)
    env.close()


def test_probe_stack_env_masks_top_source_blocks() -> None:
    env = JengaPandaProbeStackEnv()
    _, info = env.reset(seed=32)
    top_level = env.config.levels - 1
    for source_slot in range(env.config.blocks_per_level):
        source_action = top_level * env.config.blocks_per_level + source_slot
        assert not bool(info["action_mask"][source_action])
        extract_action = env._source_action_count + source_action
        assert not bool(info["action_mask"][extract_action])
    env.close()


def test_probe_stack_env_fast_mode_uses_same_action_shape() -> None:
    env = JengaPandaProbeStackEnv(fast_dynamics=True)
    obs, info = env.reset(seed=33)
    assert env.observation_space.contains(obs)
    assert info["physics_mode"] == "fast_force"
    assert env.action_space.n == 2 * env.config.levels * env.config.blocks_per_level
    assert info["action_mask"].shape == (env.action_space.n,)
    env.close()
