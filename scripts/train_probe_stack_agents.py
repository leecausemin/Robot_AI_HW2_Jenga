#!/usr/bin/env python3
"""Train PPO or Maskable PPO on the active-probing Jenga environment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import time

import numpy as np

from jenga_rl.envs import JengaPandaProbeStackEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def make_env(
    seed: int = 0,
    *,
    fast_dynamics: bool = True,
    levels: int = 10,
    max_removed_blocks: int = 6,
    primitive_step_scale: float = 1.0,
):
    config_kwargs = {"levels": levels, "max_removed_blocks": max_removed_blocks}
    if fast_dynamics:
        config_kwargs.update({"settle_steps": 1, "motion_steps": 1, "push_steps": 1})
    config = PandaJengaConfig(**config_kwargs)
    env = JengaPandaProbeStackEnv(
        config=config,
        render_mode="rgb_array",
        fast_dynamics=fast_dynamics,
        primitive_step_scale=primitive_step_scale,
    )
    env.reset(seed=seed)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "maskable_ppo"], required=True)
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/probe_stack"))
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--max-removed-blocks", type=int, default=6)
    parser.add_argument("--full-robot", action="store_true", help="Train with visible robot motion primitives instead of fast force-push dynamics.")
    parser.add_argument("--primitive-step-scale", type=float, default=0.2, help="Scale primitive simulation steps; 0.2 keeps fast-force training practical.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out / args.algo
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        seed=args.seed,
        fast_dynamics=not args.full_robot,
        levels=args.levels,
        max_removed_blocks=args.max_removed_blocks,
        primitive_step_scale=args.primitive_step_scale,
    )

    start = time()
    if args.algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            n_steps=32,
            batch_size=16,
            n_epochs=4,
            gamma=0.95,
            learning_rate=3e-4,
            tensorboard_log=str(run_dir / "tb"),
        )
    else:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            n_steps=32,
            batch_size=16,
            n_epochs=4,
            gamma=0.95,
            learning_rate=3e-4,
            tensorboard_log=str(run_dir / "tb"),
        )

    model.learn(total_timesteps=args.timesteps)
    model_path = run_dir / "model.zip"
    model.save(model_path)
    elapsed = time() - start
    metadata = {
        "algo": args.algo,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "model_path": str(model_path),
        "env": "JengaPandaProbeStackEnv",
        "fast_dynamics": not args.full_robot,
        "levels": args.levels,
        "max_removed_blocks": args.max_removed_blocks,
        "primitive_step_scale": args.primitive_step_scale,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(metadata, indent=2))
    with (run_dir / "train_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata))
        writer.writeheader()
        writer.writerow(metadata)
    env.close()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
