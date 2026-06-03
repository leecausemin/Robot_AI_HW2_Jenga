"""Stable-Baselines3 training helpers."""

from __future__ import annotations

from pathlib import Path

from jenga_rl.envs import JengaTowerEnv


def build_model(algorithm: str, env: JengaTowerEnv, seed: int = 0, tensorboard_log: str | None = None):
    """Create an SB3 model for the discrete Jenga environment."""
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError("Install training dependencies with `pip install -e .[train]`.") from exc

    registry = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN,
    }
    key = algorithm.lower()
    if key not in registry:
        raise ValueError(f"unsupported algorithm {algorithm!r}; choose one of {sorted(registry)}")

    common_kwargs = dict(policy="MlpPolicy", env=env, seed=seed, verbose=1, tensorboard_log=tensorboard_log)
    if key == "dqn":
        return DQN(**common_kwargs, learning_starts=500, buffer_size=20_000, exploration_fraction=0.25)
    if key == "ppo":
        return PPO(**common_kwargs, n_steps=512, batch_size=64, gamma=0.98)
    return A2C(**common_kwargs, n_steps=32, gamma=0.98)


def train(algorithm: str, total_timesteps: int, seed: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = JengaTowerEnv()
    model = build_model(algorithm, env, seed=seed, tensorboard_log=str(output_dir / "tb"))
    model.learn(total_timesteps=total_timesteps)
    model_path = output_dir / f"{algorithm.lower()}_jenga"
    model.save(model_path)
    env.close()
    return model_path.with_suffix(".zip")
