"""Evaluation helpers for trained SB3 agents."""

from __future__ import annotations

import numpy as np

from jenga_rl.envs import JengaTowerEnv


def evaluate_sb3_model(model, episodes: int = 100, seed: int = 10_000) -> dict[str, float]:
    env = JengaTowerEnv()
    returns: list[float] = []
    lengths: list[int] = []
    collapses = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        total = 0.0
        length = 0
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            length += 1
        returns.append(total)
        lengths.append(length)
        collapses += int(info.get("collapsed", False))

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "collapse_rate": collapses / episodes,
    }
