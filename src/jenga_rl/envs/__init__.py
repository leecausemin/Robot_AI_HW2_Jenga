"""Environment registration for the two-robot Jenga gripper project."""

from gymnasium.envs.registration import register

from jenga_rl.envs.gripper_env import TwoRobotJengaGripperEnv
from jenga_rl.envs.panda_jenga_env import JengaPandaEnv, PandaJengaConfig

PANDA_ENV_ID = "JengaPanda-v0"
TWO_ROBOT_GRIPPER_ENV_ID = "TwoRobotJengaGripper-v0"

try:
    register(
        id=PANDA_ENV_ID,
        entry_point="jenga_rl.envs.panda_jenga_env:JengaPandaEnv",
        max_episode_steps=10,
    )
    register(
        id=TWO_ROBOT_GRIPPER_ENV_ID,
        entry_point="jenga_rl.envs.gripper_env:TwoRobotJengaGripperEnv",
        max_episode_steps=90,
    )
except Exception:
    # Gymnasium raises if the id is registered twice in an interactive session.
    pass

__all__ = [
    "PANDA_ENV_ID",
    "TWO_ROBOT_GRIPPER_ENV_ID",
    "JengaPandaEnv",
    "PandaJengaConfig",
    "TwoRobotJengaGripperEnv",
]
