# Two-Robot Jenga RL

PyBullet 기반 3D Jenga 강화학습 프로젝트입니다. 최신 메인 환경은 **두 대의 Franka Panda 로봇이 협력해서 target block 하나를 안전하게 추출하는 환경**입니다.

- **Robot A**: scripted pusher. target block을 살짝 밀어 노출합니다.
- **Robot B**: PPO agent. custom Jenga gripper로 노출된 끝부분을 잡고 블록을 바닥으로 떨어뜨립니다.
- **Main env**: `TwoRobotJengaGripperEnv`
- **Algorithm**: Stable-Baselines3 PPO (`MlpPolicy`)
- **Observation**: Robot B joint state, end-effector, target block, exposed grasp site, pull direction, stability metrics 등 36D
- **Action**: Robot B 7D joint residual + 1D gripper close command

자세한 발표용 설명은 [`PROJECT.md`](PROJECT.md)를 보세요.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[train,dev]
```

현재 작업 환경에서는 기존 venv를 사용할 수 있습니다.

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python -m pytest -q
```

## 학습

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/train_two_robot_gripper_ppo.py \
  --timesteps 8000 \
  --out runs/two_robot_gripper_ppo \
  --levels 6 \
  --max-episode-steps 90
```

## GIF 기록

### PPO 성공 rollout

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_two_robot_gripper_rollout.py \
  --policy ppo \
  --model runs/two_robot_gripper_precise_pusher_ft3k/model.zip \
  --seed 237 \
  --out-dir docs/training_media/two_robot_gripper_far_b_exposed_tip \
  --levels 6 \
  --max-episode-steps 90
```

### Untrained 실패 rollout

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_two_robot_gripper_rollout.py \
  --policy untrained \
  --seed 217 \
  --out-dir docs/training_media/two_robot_gripper_far_b_exposed_tip \
  --levels 6 \
  --max-episode-steps 90
```

## 발표용 결과물

```text
docs/training_media/two_robot_gripper_far_b_exposed_tip/ppo/rollout_seed_237.gif
docs/training_media/two_robot_gripper_far_b_exposed_tip/untrained/rollout_seed_217.gif
docs/training_media/two_robot_gripper_far_b_exposed_tip/stills/robot_b_closeup.png
```

## 코드 구조

```text
src/jenga_rl/envs/config.py                    # Jenga tower config base
src/jenga_rl/envs/panda_jenga_env.py           # PyBullet/Panda/Jenga base environment
src/jenga_rl/envs/two_robot_gripper_env.py     # latest main RL environment
scripts/train_two_robot_gripper_ppo.py         # PPO training
scripts/record_two_robot_gripper_rollout.py    # GIF/still rollout recording
tests/test_two_robot_gripper_env.py            # smoke/regression test
PROJECT.md                                     # presentation-ready project explanation
```
