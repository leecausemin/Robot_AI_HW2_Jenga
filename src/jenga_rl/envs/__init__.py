"""Environment registration for the Jenga RL project."""

from gymnasium.envs.registration import register

from jenga_rl.envs.jenga_env import JengaTowerEnv
from jenga_rl.envs.panda_jenga_env import JengaPandaEnv
from jenga_rl.envs.panda_stack_env import JengaPandaStackEnv
from jenga_rl.envs.panda_probe_stack_env import JengaPandaProbeStackEnv
from jenga_rl.envs.single_block_joint_push_env import SingleBlockJointPushEnv
from jenga_rl.envs.two_robot_gripper_env import TwoRobotJengaGripperEnv
from jenga_rl.envs.bullet_env import JengaBulletEnv

ENV_ID = "JengaTower-v0"
BULLET_ENV_ID = "JengaBullet-v0"
PANDA_ENV_ID = "JengaPanda-v0"
STACK_ENV_ID = "JengaPandaStack-v0"
PROBE_STACK_ENV_ID = "JengaPandaProbeStack-v0"
JOINT_PUSH_ENV_ID = "SingleBlockJointPush-v0"
TWO_ROBOT_GRIPPER_ENV_ID = "TwoRobotJengaGripper-v0"

try:
    register(
        id=ENV_ID,
        entry_point="jenga_rl.envs.jenga_env:JengaTowerEnv",
        max_episode_steps=18,
    )
    register(
        id=BULLET_ENV_ID,
        entry_point="jenga_rl.envs.bullet_env:JengaBulletEnv",
        max_episode_steps=12,
    )
    register(
        id=PANDA_ENV_ID,
        entry_point="jenga_rl.envs.panda_jenga_env:JengaPandaEnv",
        max_episode_steps=10,
    )
    register(
        id=STACK_ENV_ID,
        entry_point="jenga_rl.envs.panda_stack_env:JengaPandaStackEnv",
        max_episode_steps=8,
    )
    register(
        id=PROBE_STACK_ENV_ID,
        entry_point="jenga_rl.envs.panda_probe_stack_env:JengaPandaProbeStackEnv",
        max_episode_steps=12,
    )
    register(
        id=JOINT_PUSH_ENV_ID,
        entry_point="jenga_rl.envs.single_block_joint_push_env:SingleBlockJointPushEnv",
        max_episode_steps=80,
    )
    register(
        id=TWO_ROBOT_GRIPPER_ENV_ID,
        entry_point="jenga_rl.envs.two_robot_gripper_env:TwoRobotJengaGripperEnv",
        max_episode_steps=90,
    )
except Exception:
    # Gymnasium raises if the id is registered twice in an interactive session.
    pass

__all__ = ["BULLET_ENV_ID", "ENV_ID", "PANDA_ENV_ID", "PROBE_STACK_ENV_ID", "STACK_ENV_ID", "JengaBulletEnv", "JengaPandaEnv", "JengaPandaProbeStackEnv", "JengaPandaStackEnv", "JengaTowerEnv", "SingleBlockJointPushEnv", "JOINT_PUSH_ENV_ID", "TwoRobotJengaGripperEnv", "TWO_ROBOT_GRIPPER_ENV_ID"]
