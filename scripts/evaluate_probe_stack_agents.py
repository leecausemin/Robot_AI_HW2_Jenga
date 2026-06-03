#!/usr/bin/env python3
"""Evaluate policies on the multi-turn active-probing Jenga task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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


def load_model(algo: str, model_path: Path, env):
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env)
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(model_path, env=env)
    return None


def choose_action(policy: str, model: Any, obs: np.ndarray, info: dict[str, Any], env) -> int:
    legal = np.asarray(info["legal_actions"], dtype=np.int64)
    if policy == "random":
        return int(env.np_random.choice(legal))
    if policy == "greedy_probe":
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


def evaluate_policy(
    policy: str,
    model_path: Path | None,
    episodes: int,
    seed: int,
    max_steps: int,
    *,
    fast_dynamics: bool,
    levels: int,
    max_removed_blocks: int,
    primitive_step_scale: float,
) -> dict[str, float | str]:
    env = make_env(
        seed=seed,
        fast_dynamics=fast_dynamics,
        levels=levels,
        max_removed_blocks=max_removed_blocks,
        primitive_step_scale=primitive_step_scale,
    )
    model = load_model(policy, model_path, env) if model_path else None
    returns: list[float] = []
    survived_turns: list[int] = []
    episode_lengths: list[int] = []
    collapses = illegal = probes = extractions = completed_episodes = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        steps = 0
        turn_count = 0
        collapsed_this_episode = False
        while not done and steps < max_steps:
            if np.asarray(info["legal_actions"], dtype=np.int64).size == 0:
                break
            action = choose_action(policy, model, obs, info, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            illegal += int(info.get("invalid_action", False))
            probes += int(info.get("probed", False))
            extractions += int(info.get("extracted", False))
            turn_count = int(info.get("removed_count", turn_count))
            if info.get("collapsed", False):
                collapsed_this_episode = True
            if info.get("completed", False):
                completed_episodes += 1
            done = terminated or truncated
        returns.append(total)
        survived_turns.append(turn_count)
        episode_lengths.append(steps)
        collapses += int(collapsed_this_episode)

    env.close()
    target_turns = env.config.max_removed_blocks
    return {
        "policy": policy,
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_survived_turns": float(np.mean(survived_turns)),
        "std_survived_turns": float(np.std(survived_turns)),
        "max_survived_turns": float(np.max(survived_turns)),
        "survival_success_rate": float(np.mean(np.asarray(survived_turns) >= target_turns)),
        "completion_rate": completed_episodes / episodes,
        "extraction_rate": extractions / episodes,
        "collapse_rate": collapses / episodes,
        "illegal_action_rate": illegal / max(1, episodes),
        "avg_probe_count": probes / episodes,
        "avg_episode_steps": float(np.mean(episode_lengths)),
        "target_turns": float(target_turns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--runs", type=Path, default=Path("runs/probe_stack"))
    parser.add_argument("--out", type=Path, default=Path("docs/training_media/metrics/comparison.csv"))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--max-removed-blocks", type=int, default=6)
    parser.add_argument("--full-robot", action="store_true")
    parser.add_argument("--primitive-step-scale", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_kwargs = {
        "fast_dynamics": not args.full_robot,
        "levels": args.levels,
        "max_removed_blocks": args.max_removed_blocks,
        "primitive_step_scale": args.primitive_step_scale,
    }
    rows = [
        evaluate_policy("random", None, args.episodes, args.seed, args.max_steps, **eval_kwargs),
        evaluate_policy("greedy_probe", None, args.episodes, args.seed + 1000, args.max_steps, **eval_kwargs),
    ]
    for algo in ["ppo", "maskable_ppo"]:
        model_path = args.runs / algo / "model.zip"
        if model_path.exists():
            rows.append(evaluate_policy(algo, model_path, args.episodes, args.seed + 2000, args.max_steps, **eval_kwargs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
