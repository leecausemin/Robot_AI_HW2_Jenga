"""Environment registration for the two-robot Jenga gripper project."""

from gymnasium.envs.registration import register

from jenga_rl.envs.panda_jenga_env import JengaPandaEnv, PandaJengaConfig
from jenga_rl.envs.two_robot_gripper_env import TwoRobotJengaGripperEnv

TWO_ROBOT_GRIPPER_ENV_ID = "TwoRobotJengaGripper-v0"

try:
    register(
        id=TWO_ROBOT_GRIPPER_ENV_ID,
        entry_point="jenga_rl.envs.two_robot_gripper_env:TwoRobotJengaGripperEnv",
        max_episode_steps=90,
    )
except Exception:
    # Gymnasium raises if the id is registered twice in an interactive session.
    pass

__all__ = [
    "JengaPandaEnv",
    "PandaJengaConfig",
    "TwoRobotJengaGripperEnv",
    "TWO_ROBOT_GRIPPER_ENV_ID",
]
