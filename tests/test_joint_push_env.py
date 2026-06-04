from __future__ import annotations

import numpy as np

from jenga_rl.envs import SingleBlockJointPushEnv


def test_joint_push_env_smoke() -> None:
    env = SingleBlockJointPushEnv(render_mode="rgb_array", max_episode_steps=5)
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert info["task"] == "single_block_joint_push"
    assert info["target_block"] is not None
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert "target_displacement" in info
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 960, 3)
    env.close()
