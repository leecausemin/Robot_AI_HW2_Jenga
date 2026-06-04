#!/usr/bin/env python3
"""Run the trained two-robot gripper policy in the PyBullet simulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from jenga_rl.envs import TwoRobotJengaGripperEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


DEFAULT_MODEL = Path("runs/two_robot_gripper_precise_pusher_ft3k/model.zip")


def make_env(
    *,
    seed: int,
    levels: int,
    max_episode_steps: int,
    render_mode: str,
) -> tuple[TwoRobotJengaGripperEnv, np.ndarray, dict[str, Any]]:
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
    env = TwoRobotJengaGripperEnv(
        config=config,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed + 999)
    return env, obs, info


def choose_action(
    *,
    policy: str,
    model: PPO | None,
    env: TwoRobotJengaGripperEnv,
    obs: np.ndarray,
) -> np.ndarray:
    if policy == "ppo":
        assert model is not None
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    if policy == "close_baseline":
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[-1] = 1.0
        return action
    return np.asarray(env.action_space.sample(), dtype=np.float32)


def event_row(step: int, reward: float, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step,
        "reward": round(float(reward), 4),
        "completed": bool(info.get("completed", False)),
        "collapsed": bool(info.get("collapsed", False)),
        "extracted": bool(info.get("extracted", False)),
        "floor_dropped": bool(info.get("floor_dropped", False)),
        "grasped": bool(info.get("grasped", False)),
        "target_displacement": round(float(info.get("target_displacement", 0.0)), 4),
        "neighbor_motion": round(float(info.get("neighbor_motion", 0.0)), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ppo", "untrained", "close_baseline"], default="ppo")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=237)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=90)
    parser.add_argument(
        "--render-mode",
        choices=["human", "ansi", "rgb_array"],
        default="human",
        help="human opens the PyBullet GUI; ansi/rgb_array run headless.",
    )
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between policy steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.policy == "ppo" and not args.model.exists():
        raise SystemExit(f"model checkpoint not found: {args.model}")

    env, obs, info = make_env(
        seed=args.seed,
        levels=args.levels,
        max_episode_steps=args.max_episode_steps,
        render_mode=args.render_mode,
    )
    model = PPO.load(args.model, env=env) if args.policy == "ppo" else None

    print(
        json.dumps(
            {
                "policy": args.policy,
                "model": str(args.model) if args.policy == "ppo" else None,
                "seed": args.seed,
                "target_block": info.get("target_block"),
                "render_mode": args.render_mode,
            },
            indent=2,
        )
    )

    last_info = info
    try:
        for step in range(1, args.max_episode_steps + 1):
            action = choose_action(policy=args.policy, model=model, env=env, obs=obs)
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            print(json.dumps(event_row(step, reward, info)))
            if args.render_mode == "ansi":
                print(env.render())
            if args.sleep > 0:
                time.sleep(args.sleep)
            if terminated or truncated:
                break
    finally:
        env.close()

    print(
        json.dumps(
            {
                "final_completed": bool(last_info.get("completed", False)),
                "final_collapsed": bool(last_info.get("collapsed", False)),
                "final_extracted": bool(last_info.get("extracted", False)),
                "final_floor_dropped": bool(last_info.get("floor_dropped", False)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
