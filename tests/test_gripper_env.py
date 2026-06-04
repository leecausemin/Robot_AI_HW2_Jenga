from __future__ import annotations

import numpy as np

from jenga_rl.envs import TwoRobotJengaGripperEnv


def test_gripper_env_smoke() -> None:
    env = TwoRobotJengaGripperEnv(render_mode="rgb_array", max_episode_steps=4)
    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    assert info["task"] == "two_robot_custom_gripper"
    assert info["target_block"] is not None
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[-1] = 1.0
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert "grasped" in info
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 960, 3)
    env.close()
