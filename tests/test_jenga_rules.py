from __future__ import annotations

import pytest

from jenga_rl.envs import JengaBulletEnv, JengaPandaEnv, JengaTowerEnv


@pytest.mark.parametrize("env_cls", [JengaTowerEnv, JengaBulletEnv, JengaPandaEnv])
def test_top_level_is_locked_by_jenga_rule(env_cls) -> None:
    env = env_cls()
    _, info = env.reset(seed=0)
    top_level = env.config.levels - 1
    top_actions = [top_level * env.config.blocks_per_level + slot for slot in range(3)]

    assert not set(top_actions).intersection(set(int(a) for a in info["legal_actions"]))
    assert all(not bool(info["action_mask"][action]) for action in top_actions)

    before_removed = info["removed_count"]
    _, reward, terminated, truncated, after = env.step(top_actions[0])
    assert reward < 0
    assert not terminated
    assert not truncated
    assert after["invalid_action"]
    assert after["removed_count"] == before_removed
    env.close()
