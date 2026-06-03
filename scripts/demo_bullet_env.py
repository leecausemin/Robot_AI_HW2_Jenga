#!/usr/bin/env python3
"""Run and render the 3D PyBullet Jenga environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from jenga_rl.envs import JengaBulletEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/jenga_bullet_env.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = JengaBulletEnv(render_mode="rgb_array")
    _, info = env.reset(seed=args.seed)
    print(env.render_mode, info)
    for step in range(args.steps):
        # Pick a middle-layer side block for the visual demo instead of immediately
        # destroying the bottom support layer.
        preferred_action = 4 * env.config.blocks_per_level
        legal = set(int(a) for a in info["legal_actions"])
        action = preferred_action if preferred_action in legal else int(info["legal_actions"][0])
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
