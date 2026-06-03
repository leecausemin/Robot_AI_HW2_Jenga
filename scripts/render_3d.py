#!/usr/bin/env python3
"""Render the current Jenga environment as a 3D tower image.

This is a visualizer for the abstract Gymnasium environment.  It does not add
PyBullet contact physics yet; it shows the same MDP state in a 3D form that is
clearer for reports and presentations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from jenga_rl.baselines import greedy_stability_policy
from jenga_rl.envs import JengaTowerEnv


def cuboid_faces(origin: tuple[float, float, float], size: tuple[float, float, float]):
    x, y, z = origin
    dx, dy, dz = size
    vertices = np.array(
        [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]
    )
    return [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]


def block_pose(level: int, slot: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return origin and size for one block in a standard alternating Jenga layer."""
    block_length = 3.0
    block_width = 0.82
    block_height = 0.36
    gap = 0.05
    z = level * block_height

    if level % 2 == 0:
        # Long direction along x; slots are spread along y.
        x = -block_length / 2
        y = (slot - 1) * (block_width + gap) - block_width / 2
        size = (block_length, block_width, block_height)
    else:
        # Long direction along y; slots are spread along x.
        x = (slot - 1) * (block_width + gap) - block_width / 2
        y = -block_length / 2
        size = (block_width, block_length, block_height)
    return (x, y, z), size


def render_tower(env: JengaTowerEnv, out: Path) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for level in range(env.config.levels):
        for slot in range(env.config.blocks_per_level):
            if not env.tower[level, slot]:
                continue
            origin, size = block_pose(level, slot)
            color = "#60a5fa" if level % 2 else "#f59e0b"
            poly = Poly3DCollection(
                cuboid_faces(origin, size),
                facecolors=color,
                edgecolors="#1f2937",
                linewidths=0.6,
                alpha=0.92,
            )
            ax.add_collection3d(poly)

    ax.set_title("3D JengaTowerEnv state", fontsize=16, pad=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("level height")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(0, env.config.levels * 0.38)
    ax.view_init(elev=24, azim=-52)
    ax.set_box_aspect((1, 1, 1.6))
    ax.grid(False)

    removed = env.config.levels * env.config.blocks_per_level - int(env.tower.sum())
    fig.text(
        0.05,
        0.05,
        f"removed blocks: {removed}\nstability margin: {env.last_margin:.3f}\nstate: 3D visualization of the same RL MDP",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("docs/jenga_tower_3d.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = JengaTowerEnv()
    env.reset(seed=args.seed)
    for _ in range(args.steps):
        action = greedy_stability_policy(env)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    render_tower(env, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
