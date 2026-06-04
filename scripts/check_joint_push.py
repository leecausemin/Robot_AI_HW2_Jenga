#!/usr/bin/env python3
"""Check whether push-only joint motion can fully extract a random Jenga block.

This is not a trained RL policy.  It uses a simple Jacobian-transpose action that
moves the Panda end-effector along the currently selected block's push axis.  The
point is to answer the modelling question: "Can a thin pusher extract the block
without directly touching other blocks?"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import pybullet as p
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires pybullet") from exc

from jenga_rl.envs import SingleBlockJointPushEnv
from jenga_rl.envs.panda_jenga_env import PandaJengaConfig


def make_env(seed: int, *, levels: int, max_episode_steps: int) -> SingleBlockJointPushEnv:
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
        render_mode=None,
        max_episode_steps=max_episode_steps,
        joint_delta_scale=0.035,
        sim_steps_per_action=5,
        near_contact_reset=True,
    )
    env.reset(seed=seed)
    return env


def jacobian_push_action(env: SingleBlockJointPushEnv, *, scale: float) -> np.ndarray:
    dofs = list(env.PANDA_ARM_JOINTS) + list(env.PANDA_FINGER_JOINTS)
    q = [p.getJointState(env._robot_id, j, physicsClientId=env._client_id)[0] for j in dofs]
    zeros = [0.0] * len(q)
    jac_t, _ = p.calculateJacobian(
        env._robot_id,
        env.PANDA_EE_LINK,
        [0, 0, 0],
        q,
        zeros,
        zeros,
        physicsClientId=env._client_id,
    )
    j_arm = np.asarray(jac_t, dtype=np.float32)[:, :7]
    direction = env._target_push_direction().astype(np.float32)
    action = j_arm.T @ direction
    max_abs = float(np.max(np.abs(action)))
    if max_abs > 1e-6:
        action = action / max_abs
    return np.clip(scale * action, -1.0, 1.0).astype(np.float32)


def non_target_tool_contacts(env: SingleBlockJointPushEnv) -> int:
    if env._tool_id is None:
        return 0
    contacts = 0
    for key, body in env._block_ids.items():
        if key == env._target_block:
            continue
        pts = p.getContactPoints(env._tool_id, body, physicsClientId=env._client_id)
        contacts += len(pts)
    return contacts


def run_episode(seed: int, *, levels: int, max_episode_steps: int, action_scale: float) -> dict[str, object]:
    env = make_env(seed, levels=levels, max_episode_steps=max_episode_steps)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    contact_steps = 0
    max_non_target_tool_contacts = 0
    last_info = info
    for step in range(max_episode_steps):
        action = jacobian_push_action(env, scale=action_scale)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        contact_steps += int(bool(info.get("contact", False)))
        max_non_target_tool_contacts = max(max_non_target_tool_contacts, non_target_tool_contacts(env))
        last_info = info
        if terminated or truncated:
            break
    row: dict[str, object] = {
        "seed": seed,
        "target_block": str(last_info.get("target_block")),
        "completed": bool(last_info.get("completed", False)),
        "extracted": bool(last_info.get("extracted", False)),
        "collapsed": bool(last_info.get("collapsed", False)),
        "steps": step + 1,
        "return": total_reward,
        "target_displacement": float(last_info.get("target_displacement", 0.0)),
        "full_extraction_distance": float(last_info.get("full_extraction_distance", 0.0)),
        "neighbor_motion": float(last_info.get("neighbor_motion", 0.0)),
        "stability_score": float(last_info.get("stability_score", 0.0)),
        "contact_steps": contact_steps,
        "max_non_target_tool_contacts": max_non_target_tool_contacts,
    }
    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--action-scale", type=float, default=0.75)
    parser.add_argument("--out", type=Path, default=Path("runs/joint_push_feasibility"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [
        run_episode(
            args.seed_start + idx,
            levels=args.levels,
            max_episode_steps=args.max_episode_steps,
            action_scale=args.action_scale,
        )
        for idx in range(args.seeds)
    ]
    summary = {
        "policy": "scripted_jacobian_push_only",
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "levels": args.levels,
        "max_episode_steps": args.max_episode_steps,
        "action_scale": args.action_scale,
        "completion_rate": float(np.mean([r["completed"] for r in rows])),
        "collapse_rate": float(np.mean([r["collapsed"] for r in rows])),
        "avg_target_displacement": float(np.mean([r["target_displacement"] for r in rows])),
        "avg_neighbor_motion": float(np.mean([r["neighbor_motion"] for r in rows])),
        "max_non_target_tool_contacts": int(max([r["max_non_target_tool_contacts"] for r in rows] or [0])),
        "rows": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
