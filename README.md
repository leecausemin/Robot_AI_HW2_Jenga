# Two-Robot Jenga RL

PyBullet 기반 3D Jenga 강화학습 프로젝트입니다. 최신 메인 환경은 **두 대의 Franka Panda 로봇이 협력해서 target block 하나를 안전하게 추출하는 환경**입니다.

- **Robot A**: scripted pusher. target block을 살짝 밀어 노출합니다.
- **Robot B**: PPO agent. custom Jenga gripper로 노출된 끝부분을 잡고 블록을 바닥으로 떨어뜨립니다.
- **Main env**: `TwoRobotJengaGripperEnv`
- **Main file**: `src/jenga_rl/envs/gripper_env.py`
- **Algorithm**: Stable-Baselines3 PPO (`MlpPolicy`)
- **Observation**: Robot B joint state, end-effector, target block, exposed grasp site, pull direction, stability metrics 등 36D
- **Action**: Robot B 7D joint residual + 1D gripper close command

## Before / After

### Before: untrained policy

학습 전 policy는 gripper가 target block에 안정적으로 정렬하지 못하고, 블록을 제대로 잡거나 추출하지 못합니다.

![Before: untrained rollout](docs/training_media/two_robot_gripper_far_b_exposed_tip/untrained/rollout_seed_217.gif)

### After: PPO policy

PPO policy는 Robot A가 노출한 target block 끝부분으로 Robot B gripper를 이동시키고, 블록을 잡아 tower 밖으로 끌어냅니다.

![After: PPO rollout](docs/training_media/two_robot_gripper_far_b_exposed_tip/ppo/rollout_seed_237.gif)

## Simulation stills

| Untrained start | Untrained failure | Robot B gripper close-up |
| --- | --- | --- |
| ![Untrained start](docs/training_media/two_robot_gripper_far_b_exposed_tip/stills/robot_b_untrained_start.png) | ![Untrained mid/failure](docs/training_media/two_robot_gripper_far_b_exposed_tip/stills/untrained_seed217_bad_mid.png) | ![Robot B close-up](docs/training_media/two_robot_gripper_far_b_exposed_tip/stills/robot_b_closeup.png) |

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

## 실제 시뮬레이션 실행 커맨드

아래 커맨드는 repo root에서 실행합니다.

### 1. 환경 smoke test

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python -m pytest -q
```

### 2. PPO 학습 실행

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/train_gripper.py \
  --timesteps 8000 \
  --out runs/two_robot_gripper_ppo \
  --levels 6 \
  --max-episode-steps 90
```

### 3. 학습 전 untrained rollout GIF 생성

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_gripper.py \
  --policy untrained \
  --seed 217 \
  --out-dir docs/training_media/two_robot_gripper_far_b_exposed_tip \
  --levels 6 \
  --max-episode-steps 90
```

### 4. 학습된 PPO rollout GIF 생성

이미 있는 checkpoint를 사용할 때:

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_gripper.py \
  --policy ppo \
  --model runs/two_robot_gripper_precise_pusher_ft3k/model.zip \
  --seed 237 \
  --out-dir docs/training_media/two_robot_gripper_far_b_exposed_tip \
  --levels 6 \
  --max-episode-steps 90
```

방금 새로 학습한 모델을 사용할 때:

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_gripper.py \
  --policy ppo \
  --model runs/two_robot_gripper_ppo/model.zip \
  --seed 237 \
  --out-dir docs/training_media/two_robot_gripper_far_b_exposed_tip \
  --levels 6 \
  --max-episode-steps 90
```

## 발표용 결과물 경로

```text
docs/training_media/two_robot_gripper_far_b_exposed_tip/ppo/rollout_seed_237.gif
docs/training_media/two_robot_gripper_far_b_exposed_tip/untrained/rollout_seed_217.gif
docs/training_media/two_robot_gripper_far_b_exposed_tip/stills/robot_b_closeup.png
docs/training_media/two_robot_gripper_far_b_exposed_tip/stills/robot_b_untrained_start.png
docs/training_media/two_robot_gripper_far_b_exposed_tip/stills/untrained_seed217_bad_mid.png
```

## 코드 구조

```text
src/jenga_rl/envs/config.py          # Jenga tower config base
src/jenga_rl/envs/panda_jenga_env.py # PyBullet/Panda/Jenga base environment
src/jenga_rl/envs/gripper_env.py     # latest main two-robot RL environment
scripts/train_gripper.py             # PPO training for main gripper env
scripts/record_gripper.py            # GIF/still rollout recording
tests/test_gripper_env.py            # smoke/regression test
```

## 파일 관리 정책

- GitHub에는 root `README.md`만 Markdown 문서로 남깁니다.
- `.omx/`, `.claude/`, 기타 로컬 agent/runtime 상태는 git에서 제외합니다.
- 실험 히스토리용 `joint_push`, `probe_stack`, `stack` entrypoint는 제거했고 최종 gripper task만 유지합니다.
