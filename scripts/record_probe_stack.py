#!/usr/bin/env python3
"""Record multi-turn rollout GIFs for presentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from jenga_rl.envs import JengaPandaProbeStackEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def load_model(policy: str, model_path: Path, env):
    if policy == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env)
    if policy == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(model_path, env=env)
    return None


def choose_action(policy: str, model: Any, obs: np.ndarray, info: dict[str, Any], env) -> int:
    legal = np.asarray(info["legal_actions"], dtype=np.int64)
    if policy == "untrained":
        return int(env.np_random.choice(legal))
    if policy == "scripted_success":
        best = info.get("best_observed_block")
        if best is not None:
            source_action = best[0] * env.config.blocks_per_level + best[1]
            preferred = env._source_action_count + source_action
            if preferred in legal:
                return int(preferred)
        preferred_probe = (env.config.levels - 2) * env.config.blocks_per_level
        if preferred_probe in legal:
            return int(preferred_probe)
        probe_actions = legal[legal < env._source_action_count]
        if probe_actions.size:
            return int(probe_actions[0])
        return int(legal[0])
    if policy == "maskable_ppo":
        action, _ = model.predict(obs, deterministic=True, action_masks=info["action_mask"])
        return int(action)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["untrained", "scripted_success", "ppo", "maskable_ppo"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/training_media"))
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--fast-dynamics", action="store_true", help="Use the same fast task dynamics used for RL training/evaluation.")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--max-removed-blocks", type=int, default=6)
    parser.add_argument("--primitive-step-scale", type=float, default=1.0)
    parser.add_argument("--record-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.out_dir / args.policy
    frames_dir = target_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs = {"levels": args.levels, "max_removed_blocks": args.max_removed_blocks}
    if args.fast_dynamics:
        config_kwargs.update({"settle_steps": 1, "motion_steps": 1, "push_steps": 1})
    else:
        # Presentation-only placement: move the Panda slightly behind the tower
        # so the bulky wrist/gripper stays outside the stack and only the long
        # blue Jenga probe visually reaches the selected block.
        config_kwargs.update({
            "robot_base_x": -0.12,
            "robot_base_y": -0.24,
            # Presentation rollouts use stricter collapse thresholds so rough
            # pushes visibly fail instead of looking like the robot can pass
            # through the tower without consequence.
            "lateral_collapse_threshold": 0.18,
            "tilt_collapse_threshold": 0.40,
        })
    config = PandaJengaConfig(**config_kwargs)
    env = JengaPandaProbeStackEnv(
        config=config,
        render_mode="rgb_array",
        fast_dynamics=args.fast_dynamics,
        primitive_step_scale=args.primitive_step_scale,
    )
    model = load_model(args.policy, args.model, env) if args.model else None
    obs, info = env.reset(seed=args.seed)
    frames = [env.render()]
    env.enable_recording(every=args.record_every)
    events = []
    turn_count = 0
    done = False
    for step in range(args.max_steps):
        if np.asarray(info["legal_actions"], dtype=np.int64).size == 0:
            break
        action = choose_action(args.policy, model, obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        motion_frames = env.drain_frames()
        if motion_frames:
            frames.extend(motion_frames)
        else:
            frames.extend([env.render()] * 6)
        turn_count = int(info.get("removed_count", turn_count))
        events.append(
            {
                "step": step + 1,
                "turn_count": turn_count,
                "action": int(action),
                "reward": float(reward),
                "probed": bool(info.get("probed", False)),
                "extracted": bool(info.get("extracted", False)),
                "collapsed": bool(info.get("collapsed", False)),
                "completed": bool(info.get("completed", False)),
            }
        )
        done = terminated or truncated
        if done:
            break

    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"frame_{idx:03d}.png")
    gif_path = target_dir / "rollout.gif"
    imageio.mimsave(gif_path, frames, duration=0.08)
    summary = {
        "policy": args.policy,
        "seed": args.seed,
        "gif": str(gif_path),
        "survived_turns": turn_count,
        "target_turns": env.config.max_removed_blocks,
        "events": events,
        "final_info": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v for k, v in info.items() if k not in {"legal_actions", "action_mask"}},
    }
    (target_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
