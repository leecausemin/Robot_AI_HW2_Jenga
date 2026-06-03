#!/usr/bin/env python3
"""Render one full Jenga turn: extract a block and place it on top."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from jenga_rl.envs import JengaPandaStackEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/jenga_panda_stack_after_turn.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = JengaPandaStackEnv(render_mode="rgb_array")
    _, info = env.reset(seed=args.seed)
    print("reset", {k: v for k, v in info.items() if k not in {"legal_actions", "action_mask"}})
    for step in range(args.steps):
        # Demonstration action: upper non-top source block, top placement slot 1.
        source_level = env.config.levels - 2
        extraction_action = source_level * env.config.blocks_per_level
        action = extraction_action * 3 + 1
        if action not in set(int(a) for a in info["legal_actions"]):
            action = int(info["legal_actions"][0])
        _, reward, terminated, truncated, info = env.step(action)
        print(f"step={step + 1} action={action} reward={reward:.3f}", {k: v for k, v in info.items() if k not in {"legal_actions", "action_mask"}})
        if terminated or truncated:
            break
    frame = env.render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(args.out)
    print(env.render_mode, args.out)
    env.close()


if __name__ == "__main__":
    main()
