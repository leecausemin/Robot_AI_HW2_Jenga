#!/usr/bin/env python3
"""Train one Stable-Baselines3 agent on JengaTower-v0."""

from __future__ import annotations

import argparse
from pathlib import Path

from jenga_rl.training.sb3_runner import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = train(args.algo, args.timesteps, args.seed, args.out / args.algo)
    print(f"saved model: {path}")


if __name__ == "__main__":
    main()
