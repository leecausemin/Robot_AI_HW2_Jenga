#!/usr/bin/env python3
"""Check the custom environment against Gymnasium/SB3 expectations."""

from __future__ import annotations

from jenga_rl.envs import JengaTowerEnv


def main() -> None:
    env = JengaTowerEnv()
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert len(info["legal_actions"]) > 0

    for _ in range(5):
        action = int(info["legal_actions"][0])
        obs, _, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break

    try:
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("basic check passed; install `.[train]` to run SB3 check_env")
    else:
        check_env(JengaTowerEnv(), warn=True)
        print("SB3 check_env passed")


if __name__ == "__main__":
    main()
