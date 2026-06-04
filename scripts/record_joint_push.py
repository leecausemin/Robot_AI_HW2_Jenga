#!/usr/bin/env python3
"""Record a SingleBlockJointPushEnv rollout GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from jenga_rl.envs import SingleBlockJointPushEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def make_env(seed: int, levels: int, max_episode_steps: int):
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
    env = SingleBlockJointPushEnv(config=config, render_mode="rgb_array", max_episode_steps=max_episode_steps, sim_steps_per_action=5)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed + 999)
    return env, obs, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ppo", "untrained", "gentle_scripted"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/training_media/joint_push"))
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--record-every", type=int, default=4)
    parser.add_argument(
        "--settle-after-done",
        type=int,
        default=80,
        help="Extra PyBullet steps to record after success/failure, so a fully extracted block can slide/fall visibly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.out_dir / args.policy
    frames_dir = target_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    env, obs, info = make_env(args.seed, args.levels, args.max_episode_steps)
    model = PPO.load(args.model, env=env) if args.policy == "ppo" and args.model else None
    frames = [env.render()]
    env.enable_recording(every=args.record_every)
    events: list[dict[str, Any]] = []
    done = False
    last_info = info
    for step in range(args.max_episode_steps):
        if args.policy == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif args.policy == "gentle_scripted":
            # small positive movement on all joints; only for sanity contrast, not used as result.
            action = np.asarray([0.15, 0.0, 0.05, 0.0, 0.0, 0.05, 0.0], dtype=np.float32)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        motion_frames = env.drain_frames()
        frames.extend(motion_frames if motion_frames else [env.render()] * 3)
        events.append({
            "step": step + 1,
            "reward": float(reward),
            "target_displacement": float(info.get("target_displacement", 0.0)),
            "neighbor_motion": float(info.get("neighbor_motion", 0.0)),
            "contact": bool(info.get("contact", False)),
            "extracted": bool(info.get("extracted", False)),
            "collapsed": bool(info.get("collapsed", False)),
            "completed": bool(info.get("completed", False)),
        })
        done = bool(terminated or truncated)
        last_info = info
        if done:
            break
    if done and args.settle_after_done > 0:
        for _ in range(args.settle_after_done):
            env._step_simulation(1)
            motion_frames = env.drain_frames()
            frames.extend(motion_frames if motion_frames else [])
        if not frames or len(frames) < 2:
            frames.append(env.render())
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"frame_{idx:03d}.png")
    gif_path = target_dir / "rollout.gif"
    imageio.mimsave(gif_path, frames, duration=0.08)
    summary = {
        "policy": args.policy,
        "seed": args.seed,
        "gif": str(gif_path),
        "events": events,
        "final_info": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v for k, v in last_info.items()},
    }
    (target_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
