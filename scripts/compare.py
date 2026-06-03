#!/usr/bin/env python3
"""Run baseline comparisons, and optionally train SB3 algorithms."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jenga_rl.baselines import evaluate_policy, greedy_stability_policy, random_policy
from jenga_rl.envs import JengaTowerEnv
from jenga_rl.evaluation.sb3_eval import evaluate_sb3_model
from jenga_rl.training.q_learning import QLearningConfig, evaluate_q_learning, train_q_learning
from jenga_rl.training.sb3_runner import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--q-episodes", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=0, help="0 skips SB3 training")
    parser.add_argument("--out", type=Path, default=Path("runs/comparison.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | str]] = []
    env = JengaTowerEnv()
    for name, policy in [("random", random_policy), ("greedy_stability", greedy_stability_policy)]:
        metrics = evaluate_policy(env, policy, episodes=args.episodes)
        rows.append({"agent": name, **metrics})
        print(name, metrics)

    if args.q_episodes > 0:
        learner, _ = train_q_learning(QLearningConfig(episodes=args.q_episodes))
        metrics = evaluate_q_learning(learner, episodes=args.episodes)
        rows.append({"agent": "tabular_q_learning", **metrics})
        print("tabular_q_learning", metrics)

    if args.timesteps > 0:
        for algo in ["ppo", "a2c", "dqn"]:
            train_env = JengaTowerEnv()
            model = build_model(algo, train_env, seed=0)
            model.learn(total_timesteps=args.timesteps)
            metrics = evaluate_sb3_model(model, episodes=args.episodes)
            rows.append({"agent": algo, **metrics})
            print(algo, metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
