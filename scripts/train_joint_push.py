#!/usr/bin/env python3
"""Train PPO for single-block Panda joint-control Jenga pushing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from jenga_rl.envs import SingleBlockJointPushEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def make_env(seed: int, *, levels: int, max_episode_steps: int, render_mode: str | None = None):
    config = PandaJengaConfig(
        levels=levels,
        max_removed_blocks=1,
        settle_steps=40,
        motion_steps=35,
        push_steps=50,
        robot_base_x=-0.12,
        robot_base_y=-0.24,
        lateral_collapse_threshold=0.16,
        tilt_collapse_threshold=0.38,
        end_effector="jenga_probe",
    )
    env = SingleBlockJointPushEnv(
        config=config,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        joint_delta_scale=0.035,
        sim_steps_per_action=5,
        near_contact_reset=True,
    )
    env.reset(seed=seed)
    return Monitor(env)


def evaluate(model: PPO, *, seed: int, episodes: int, levels: int, max_episode_steps: int) -> dict[str, float]:
    returns = []
    completed = []
    collapsed = []
    extracted = []
    steps = []
    target_disp = []
    neighbor = []
    contacts = []
    for ep in range(episodes):
        env = make_env(seed + ep, levels=levels, max_episode_steps=max_episode_steps)
        obs, info = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        step_count = 0
        last_info = info
        contact_count = 0
        while not done and step_count < max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            step_count += 1
            contact_count += int(bool(info.get("contact", False)))
            done = bool(terminated or truncated)
            last_info = info
        returns.append(total)
        completed.append(float(last_info.get("completed", False)))
        collapsed.append(float(last_info.get("collapsed", False)))
        extracted.append(float(last_info.get("extracted", False)))
        steps.append(float(step_count))
        target_disp.append(float(last_info.get("target_displacement", 0.0)))
        neighbor.append(float(last_info.get("neighbor_motion", 0.0)))
        contacts.append(float(contact_count))
        env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "completion_rate": float(np.mean(completed)),
        "collapse_rate": float(np.mean(collapsed)),
        "extraction_rate": float(np.mean(extracted)),
        "avg_steps": float(np.mean(steps)),
        "avg_target_displacement": float(np.mean(target_disp)),
        "avg_neighbor_motion": float(np.mean(neighbor)),
        "avg_contact_steps": float(np.mean(contacts)),
    }


class EarlyStopEvalCallback(BaseCallback):
    def __init__(self, *, out_dir: Path, eval_every: int, seed: int, levels: int, max_episode_steps: int, threshold: float):
        super().__init__()
        self.out_dir = out_dir
        self.eval_every = eval_every
        self.seed = seed
        self.levels = levels
        self.max_episode_steps = max_episode_steps
        self.threshold = threshold
        self.rows: list[dict[str, float]] = []
        self.best_completion = -1.0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.eval_every or self.num_timesteps % self.eval_every != 0:
            return True
        metrics = evaluate(
            self.model,
            seed=self.seed + 10000 + self.num_timesteps,
            episodes=5,
            levels=self.levels,
            max_episode_steps=self.max_episode_steps,
        )
        row = {"timesteps": float(self.num_timesteps), **metrics}
        self.rows.append(row)
        print(json.dumps(row, indent=2))
        if metrics["completion_rate"] > self.best_completion:
            self.best_completion = metrics["completion_rate"]
            self.model.save(self.out_dir / "best_model.zip")
        self._write_rows()
        return metrics["completion_rate"] < self.threshold

    def _write_rows(self) -> None:
        if not self.rows:
            return
        path = self.out_dir / "eval_history.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (self.out_dir / "eval_history.json").write_text(json.dumps(self.rows, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/joint_push_ppo"))
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--early-stop-completion", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.seed, levels=args.levels, max_episode_steps=args.max_episode_steps)
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
        tensorboard_log=str(run_dir / "tb"),
    )
    callback = EarlyStopEvalCallback(
        out_dir=run_dir,
        eval_every=args.eval_every,
        seed=args.seed,
        levels=args.levels,
        max_episode_steps=args.max_episode_steps,
        threshold=args.early_stop_completion,
    )
    start = time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model_path = run_dir / "model.zip"
    model.save(model_path)
    final_metrics = evaluate(model, seed=args.seed + 20000, episodes=10, levels=args.levels, max_episode_steps=args.max_episode_steps)
    summary = {
        "algo": "ppo",
        "env": "SingleBlockJointPushEnv",
        "timesteps_requested": args.timesteps,
        "timesteps_done": model.num_timesteps,
        "elapsed_sec": time() - start,
        "model_path": str(model_path),
        "best_model_path": str(run_dir / "best_model.zip"),
        "levels": args.levels,
        "max_episode_steps": args.max_episode_steps,
        **final_metrics,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    with (run_dir / "train_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
