#!/usr/bin/env python3
"""Record rollout GIFs for the two-robot custom-gripper Jenga task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from jenga_rl.envs import TwoRobotJengaGripperEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def make_env(seed: int, levels: int, max_episode_steps: int, record_every: int) -> tuple[TwoRobotJengaGripperEnv, np.ndarray, dict[str, Any]]:
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
    env = TwoRobotJengaGripperEnv(config=config, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    env.enable_recording(every=record_every)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed + 999)
    return env, obs, info


def choose_action(policy: str, model: PPO | None, env: TwoRobotJengaGripperEnv, obs: np.ndarray) -> np.ndarray:
    if policy == "ppo":
        assert model is not None
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    if policy == "close_baseline":
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[-1] = 1.0
        return action
    return np.asarray(env.action_space.sample(), dtype=np.float32)


def serializable_info(info: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, tuple):
            clean[key] = list(value)
        else:
            clean[key] = str(value)
    return clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ppo", "untrained", "close_baseline"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/training_media/two_robot_gripper"))
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=90)
    parser.add_argument("--record-every", type=int, default=4)
    parser.add_argument("--settle-after-done", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.policy == "ppo" and args.model is None:
        raise SystemExit("--model is required for --policy ppo")
    target_dir = args.out_dir / args.policy
    frames_dir = target_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env, obs, info = make_env(args.seed, args.levels, args.max_episode_steps, args.record_every)
    model = PPO.load(args.model, env=env) if args.policy == "ppo" else None

    frames = env.drain_frames()
    frames.append(env.render())
    events: list[dict[str, Any]] = []
    last_info = info
    for step in range(args.max_episode_steps):
        action = choose_action(args.policy, model, env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        motion_frames = env.drain_frames()
        frames.extend(motion_frames if motion_frames else [env.render()] * 3)
        events.append(
            {
                "step": step + 1,
                "reward": float(reward),
                "completed": bool(info.get("completed", False)),
                "collapsed": bool(info.get("collapsed", False)),
                "floor_dropped": bool(info.get("floor_dropped", False)),
                "extracted": bool(info.get("extracted", False)),
                "grasped": bool(info.get("grasped", False)),
                "ever_grasped": bool(info.get("ever_grasped", False)),
                "target_displacement": float(info.get("target_displacement", 0.0)),
                "target_height": float(info.get("target_height", 0.0)),
                "neighbor_motion": float(info.get("neighbor_motion", 0.0)),
                "non_target_gripper_contacts": int(info.get("non_target_gripper_contacts", 0)),
            }
        )
        last_info = info
        if terminated or truncated:
            break

    if args.settle_after_done > 0:
        for _ in range(args.settle_after_done):
            env._step_simulation(1)
        frames.extend(env.drain_frames())
        frames.append(env.render())

    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"frame_{idx:03d}.png")
    gif_path = target_dir / f"rollout_seed_{args.seed}.gif"
    imageio.mimsave(gif_path, frames, duration=0.08)
    summary = {
        "policy": args.policy,
        "seed": args.seed,
        "gif": str(gif_path),
        "frame_count": len(frames),
        "events": events,
        "final_info": serializable_info(last_info),
    }
    (target_dir / f"summary_seed_{args.seed}.json").write_text(json.dumps(summary, indent=2))
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
