from __future__ import annotations

import numpy as np

from jenga_rl.envs import JengaTowerEnv


def test_reset_observation_is_valid() -> None:
    env = JengaTowerEnv()
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert info["removed_count"] == 0
    assert info["legal_actions"].size > 0


def test_legal_action_removes_one_block() -> None:
    env = JengaTowerEnv()
    _, info = env.reset(seed=1)
    action = int(info["legal_actions"][0])
    before = env.tower.sum()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert env.tower.sum() == before - 1
    assert isinstance(reward, float)
    assert not truncated
    assert terminated or info.get("success") or info.get("collapsed")


def test_invalid_top_action_does_not_mutate_tower() -> None:
    env = JengaTowerEnv()
    env.reset(seed=1)
    top_action = (env.config.levels - 1) * env.config.blocks_per_level
    before = env.tower.copy()
    _, reward, terminated, truncated, info = env.step(top_action)
    assert np.array_equal(env.tower, before)
    assert reward < 0
    assert not terminated
    assert not truncated
    assert info["invalid_action"]


def test_greedy_margin_simulation_is_read_only() -> None:
    env = JengaTowerEnv()
    _, info = env.reset(seed=7)
    before = env.tower.copy()
    margin = env.simulate_action_margin(int(info["legal_actions"][0]))
    assert margin >= -1.0
    assert np.array_equal(env.tower, before)
