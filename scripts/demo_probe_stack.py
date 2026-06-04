#!/usr/bin/env python3
"""Render active probing followed by extraction and top placement."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from jenga_rl.envs import JengaPandaProbeStackEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/probe_stack_turn.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = JengaPandaProbeStackEnv(render_mode="rgb_array")
    _, info = env.reset(seed=args.seed)
    print("reset", {k: v for k, v in info.items() if k not in {"legal_actions", "action_mask"}})

    # Demo: probe a reachable upper non-top side block, then extract/place it.
    source_level = env.config.levels - 2
    source_action = source_level * env.config.blocks_per_level
    probe_action = source_action
    if probe_action not in set(int(a) for a in info["legal_actions"]):
        probe_action = int(info["legal_actions"][0])
        source_action = probe_action

    _, reward, terminated, truncated, info = env.step(probe_action)
    print(f"probe action={probe_action} reward={reward:.3f}", {k: v for k, v in info.items() if k not in {"legal_actions", "action_mask"}})

    place_slot = 1
    extract_action = env._source_action_count + source_action * 3 + place_slot
    if extract_action not in set(int(a) for a in info["legal_actions"]):
        extract_action = int(info["legal_actions"][0])
    _, reward, terminated, truncated, info = env.step(extract_action)
    print(f"extract/place action={extract_action} reward={reward:.3f}", {k: v for k, v in info.items() if k not in {"legal_actions", "action_mask"}})

    frame = env.render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(args.out)
    print(args.out)
    env.close()


if __name__ == "__main__":
    main()
