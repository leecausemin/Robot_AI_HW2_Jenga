#!/usr/bin/env python3
"""Render the Capstone-style Panda robot Jenga environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from jenga_rl.envs import JengaPandaEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/jenga_panda_env.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = JengaPandaEnv(render_mode="rgb_array")
    _, info = env.reset(seed=args.seed)
    print("reset", info)
    for step in range(args.steps):
        # Visual demo: choose a reachable middle side block.
        action = (env.config.levels - 2) * env.config.blocks_per_level
        if action not in set(int(a) for a in info["legal_actions"]):
            action = int(info["legal_actions"][0])
        _, reward, terminated, truncated, info = env.step(action)
        print(f"step={step + 1} action={action} reward={reward:.3f} info={info}")
        if terminated or truncated:
            break
    frame = env.render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(args.out)
    env.close()
    print(args.out)


if __name__ == "__main__":
    main()
