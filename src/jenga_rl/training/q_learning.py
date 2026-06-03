"""Tabular Q-learning for the Jenga environment.

This module follows the classroom progression: define an MDP, estimate an
action-value function Q(s, a), select actions with epsilon-greedy exploration,
and update the table with a Temporal-Difference target.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from jenga_rl.envs import JengaTowerEnv

StateKey = tuple[int, ...]


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 2_000
    alpha: float = 0.10
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    seed: int = 0


def state_key(observation: np.ndarray) -> StateKey:
    """Aggregate the observation into a hashable tabular state.

    The first 54 values are tower occupancy bits.  The remaining continuous
    summary values are rounded into coarse bins, which mirrors state aggregation
    from function-approximation lectures while keeping tabular Q-learning simple.
    """
    occupancy = tuple(int(x) for x in observation[:-3])
    removed_bin = int(round(float(observation[-3]) * 10))
    margin_bin = int(round(float(observation[-2]) * 10))
    legal_bin = int(round(float(observation[-1]) * 10))
    return occupancy + (removed_bin, margin_bin, legal_bin)


class TabularQLearner:
    """Dictionary-backed Q-table with epsilon-greedy action selection."""

    def __init__(self, action_count: int, rng: np.random.Generator) -> None:
        self.action_count = action_count
        self.rng = rng
        self.q: defaultdict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(action_count, dtype=np.float32)
        )

    def act(self, key: StateKey, legal_actions: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal_actions))
        values = self.q[key][legal_actions]
        return int(legal_actions[int(values.argmax())])

    def greedy_action(self, key: StateKey, legal_actions: np.ndarray) -> int:
        values = self.q[key][legal_actions]
        return int(legal_actions[int(values.argmax())])

    def update(
        self,
        key: StateKey,
        action: int,
        reward: float,
        next_key: StateKey,
        next_legal_actions: np.ndarray,
        done: bool,
        alpha: float,
        gamma: float,
    ) -> None:
        bootstrap = 0.0 if done else float(self.q[next_key][next_legal_actions].max())
        td_target = reward + gamma * bootstrap
        td_error = td_target - float(self.q[key][action])
        self.q[key][action] += alpha * td_error


def train_q_learning(config: QLearningConfig | None = None) -> tuple[TabularQLearner, list[float]]:
    cfg = config or QLearningConfig()
    rng = np.random.default_rng(cfg.seed)
    env = JengaTowerEnv()
    learner = TabularQLearner(env.action_space.n, rng)
    returns: list[float] = []

    for episode in range(cfg.episodes):
        obs, info = env.reset(seed=cfg.seed + episode)
        key = state_key(obs)
        total = 0.0
        done = False
        progress = episode / max(1, cfg.episodes - 1)
        epsilon = cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)

        while not done:
            legal = info["legal_actions"]
            action = learner.act(key, legal, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_key = state_key(next_obs)
            done = terminated or truncated
            next_legal = info["legal_actions"] if len(info["legal_actions"]) else np.arange(env.action_space.n)
            learner.update(key, action, reward, next_key, next_legal, done, cfg.alpha, cfg.gamma)
            key = next_key
            total += reward

        returns.append(total)

    env.close()
    return learner, returns


def evaluate_q_learning(
    learner: TabularQLearner,
    episodes: int = 100,
    seed: int = 20_000,
) -> dict[str, float]:
    env = JengaTowerEnv()
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    collapses = 0

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        total = 0.0
        length = 0
        done = False
        while not done:
            key = state_key(obs)
            action = learner.greedy_action(key, info["legal_actions"])
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            length += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(length)
        successes += int(info.get("success", False) and not info.get("collapsed", False))
        collapses += int(info.get("collapsed", False))

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": successes / episodes,
        "collapse_rate": collapses / episodes,
    }
