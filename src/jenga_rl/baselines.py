"""Simple non-learning baselines for the Jenga environment."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from jenga_rl.envs import JengaTowerEnv

Policy = Callable[[JengaTowerEnv], int]


def random_policy(env: JengaTowerEnv) -> int:
    legal = env.legal_actions()
    return int(env.np_random.choice(legal))


def greedy_stability_policy(env: JengaTowerEnv) -> int:
    """Choose the action with the best conservative stability/risk score.

    This is not a learning algorithm; it is a transparent baseline that mimics a
    cautious robot: prefer blocks that leave the tower stable, avoid center blocks,
    and avoid deep support layers when scores are otherwise close.
    """
    legal = env.legal_actions()
    scores = []
    for action in legal:
        level, slot = divmod(int(action), env.config.blocks_per_level)
        margin = env.simulate_action_margin(int(action))
        center_penalty = 0.12 if slot == 1 else 0.0
        depth_penalty = 0.04 * (1.0 - level / max(1, env.config.levels - 1))
        scores.append(margin - center_penalty - depth_penalty)
    return int(legal[int(np.asarray(scores).argmax())])


def evaluate_policy(env: JengaTowerEnv, policy: Policy, episodes: int = 100, seed: int = 0) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    collapses = 0

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        total_reward = 0.0
        length = 0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action = policy(env)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
        returns.append(total_reward)
        lengths.append(length)
        successes += int(info.get("success", False) and not info.get("collapsed", False))
        collapses += int(info.get("collapsed", False))

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": successes / episodes,
        "collapse_rate": collapses / episodes,
    }
