#!/usr/bin/env python3
"""Train PPO for the two-robot custom-gripper Jenga extraction task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from jenga_rl.envs import TwoRobotJengaGripperEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def make_env(seed: int, *, levels: int, max_episode_steps: int, render_mode: str | None = None):
    config = PandaJengaConfig(
        levels=levels,
        max_removed_blocks=1,
        settle_steps=40,
        motion_steps=35,
        push_steps=50,
        lateral_collapse_threshold=0.18,
        tilt_collapse_threshold=0.42,
        end_effector="none",
    )
    env = TwoRobotJengaGripperEnv(config=config, render_mode=render_mode, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    return Monitor(env)


def evaluate(model: PPO, *, seed: int, episodes: int, levels: int, max_episode_steps: int) -> dict[str, float]:
    rows = []
    for ep in range(episodes):
        env = make_env(seed + ep, levels=levels, max_episode_steps=max_episode_steps)
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        last = info
        for step in range(max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            last = info
            if terminated or truncated:
                break
        rows.append({
            "return": total,
            "completed": float(last.get("completed", False)),
            "collapsed": float(last.get("collapsed", False)),
            "extracted": float(last.get("extracted", False)),
            "floor": float(last.get("floor_dropped", False)),
            "ever_grasped": float(last.get("ever_grasped", False)),
            "steps": float(step + 1),
            "target_displacement": float(last.get("target_displacement", 0.0)),
            "target_height": float(last.get("target_height", 0.0)),
            "neighbor_motion": float(last.get("neighbor_motion", 0.0)),
            "non_target_contacts": float(last.get("non_target_gripper_contacts", 0.0)),
        })
        env.close()
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


class EvalStop(BaseCallback):
    def __init__(self, *, out_dir: Path, eval_every: int, seed: int, levels: int, max_episode_steps: int, threshold: float):
        super().__init__()
        self.out_dir = out_dir
        self.eval_every = eval_every
        self.seed = seed
        self.levels = levels
        self.max_episode_steps = max_episode_steps
        self.threshold = threshold
        self.rows: list[dict[str, float]] = []
        self.best = -1.0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.eval_every or self.num_timesteps % self.eval_every != 0:
            return True
        metrics = evaluate(self.model, seed=self.seed + 10000 + self.num_timesteps, episodes=5, levels=self.levels, max_episode_steps=self.max_episode_steps)
        row = {"timesteps": float(self.num_timesteps), **metrics}
        self.rows.append(row)
        print(json.dumps(row, indent=2))
        if metrics["completed"] > self.best:
            self.best = metrics["completed"]
            self.model.save(self.out_dir / "best_model.zip")
        self._write()
        return metrics["completed"] < self.threshold

    def _write(self) -> None:
        if not self.rows:
            return
        (self.out_dir / "eval_history.json").write_text(json.dumps(self.rows, indent=2))
        with (self.out_dir / "eval_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/two_robot_gripper_ppo"))
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=90)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--early-stop-completion", type=float, default=0.8)
    parser.add_argument("--init-model", type=Path, help="Optional PPO checkpoint to fine-tune instead of training from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.seed, levels=args.levels, max_episode_steps=args.max_episode_steps)
    if args.init_model:
        model = PPO.load(args.init_model, env=env, verbose=1, tensorboard_log=str(args.out / "tb"))
    else:
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            n_steps=128,
            batch_size=64,
            n_epochs=5,
            gamma=0.97,
            learning_rate=3e-4,
            ent_coef=0.01,
            tensorboard_log=str(args.out / "tb"),
        )
    cb = EvalStop(out_dir=args.out, eval_every=args.eval_every, seed=args.seed, levels=args.levels, max_episode_steps=args.max_episode_steps, threshold=args.early_stop_completion)
    start = time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model_path = args.out / "model.zip"
    model.save(model_path)
    metrics = evaluate(model, seed=args.seed + 20000, episodes=10, levels=args.levels, max_episode_steps=args.max_episode_steps)
    summary = {"algo": "ppo", "env": "TwoRobotJengaGripperEnv", "timesteps_requested": args.timesteps, "timesteps_done": model.num_timesteps, "elapsed_sec": time() - start, "model_path": str(model_path), "best_model_path": str(args.out / "best_model.zip"), "levels": args.levels, "max_episode_steps": args.max_episode_steps, **metrics}
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "train_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader(); writer.writerow(summary)
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
