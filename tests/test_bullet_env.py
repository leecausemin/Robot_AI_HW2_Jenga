from __future__ import annotations

import numpy as np

from jenga_rl.envs import JengaBulletEnv


def test_bullet_env_reset_step_and_rgb_render() -> None:
    env = JengaBulletEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=5)
    assert env.observation_space.contains(obs)
    assert info["physics"] == "pybullet"
    action = int(info["legal_actions"][0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert not truncated
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 960, 3)
    env.close()
