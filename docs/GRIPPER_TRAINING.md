# Two-Robot Custom Gripper Jenga Training Summary

## 목표

한 개의 목표 젠가 블록을 다른 블록이 떨어지지 않도록 조심해서 빼낸 뒤, 바닥까지 완전히 떨어뜨리는 로봇 조작 과제이다.

- Robot A: 목표 블록을 조금만 밀어서 집게가 잡을 수 있게 노출시키는 scripted pusher
- Robot B: 젠가용 커스텀 gripper를 단 Panda 로봇팔
- RL agent: Robot B의 7개 관절 residual + gripper close 명령을 제어
- 성공 조건: 목표 블록을 잡았고, 충분히 뽑았고, 바닥 높이까지 떨어졌으며, 다른 블록은 붕괴하지 않음

## 환경

메인 환경은 `TwoRobotJengaGripperEnv`이다.

- 파일: `src/jenga_rl/envs/gripper_env.py`
- Gym ID: `TwoRobotJengaGripper-v0`
- 물리 엔진: PyBullet
- Tower: 6층 젠가 타워
- Target curriculum: 빠른 실험을 위해 1층 edge block `(1, 0)` 또는 `(1, 2)` 중 랜덤 선택
- Gripper: 젠가 폭에 맞춘 커스텀 wide gripper visual + PyBullet fixed constraint grasp primitive

커스텀 gripper에서 완전한 손가락 접촉 grasp를 처음부터 RL로 학습시키면 시간이 너무 오래 걸리기 때문에, 이 과제에서는 `grasp primitive + 관절 residual RL` 구조를 사용했다. 즉, RL은 “어느 블록을 뽑을지”가 아니라, 이미 선택된 블록을 잡은 뒤 어떤 관절 움직임으로 안전하게 당기고 내려놓을지를 학습한다.

## Action / Observation / Reward

### Action

`Box(-1, 1, shape=(8,))`

1. 7D Panda arm joint residual
2. 1D gripper close command

환경 내부에는 Jacobian 기반 prior motion이 들어있고, PPO는 그 위에 residual을 더한다. 이렇게 해야 5k timestep 수준의 짧은 실험에서도 수렴 가능하다.

### Observation

Robot B 관절 상태, end-effector 위치, 목표 블록 위치, gripper-target 상대 위치, pull direction, displacement, target height, grasp/floor/drop 상태, neighbor motion 등을 포함한다.

### Reward

- 목표 블록 displacement 증가 보상
- grasp 유지 보상
- 바닥까지 떨어뜨리면 큰 성공 보상
- neighbor motion 증가, 다른 블록 접촉, collapse는 패널티

## 알고리즘

PPO를 사용했다.

- Script: `scripts/train_gripper.py`
- Model: `runs/two_robot_gripper_ppo_5k/best_model.zip`
- Timesteps: 5,120 steps
- 학습 시간: 약 450초
- Policy: MLP policy

주요 하이퍼파라미터:

```text
n_steps=128
batch_size=64
n_epochs=5
gamma=0.97
learning_rate=3e-4
ent_coef=0.01
```

## 결과

최고 eval checkpoint 기준 5 episode 평가에서 성공률 0.8을 달성했다.

학습 종료 후 10 episode 평가 요약:

```text
completed: 0.6
collapsed: 0.4
extracted: 0.8
floor_dropped: 0.6
ever_grasped: 1.0
non_target_gripper_contacts: 0.0
mean target displacement: 0.220 m
```

발표용 성공 rollout seed 206:

```text
completed=True
collapsed=False
floor_dropped=True
ever_grasped=True
target_displacement=0.270 m
target_height=0.027 m
non_target_gripper_contacts=0
```

비학습 random policy는 같은 seed 206에서 90 step 동안 grasp/extraction/floor drop 모두 실패했다.

## 발표용 자료

- 학습된 PPO 성공 GIF: `docs/training_media/two_robot_gripper/ppo/rollout_seed_206.gif`
- 비학습 random 실패 GIF: `docs/training_media/two_robot_gripper/untrained/rollout_seed_206.gif`
- 성공 장면 stills: `docs/training_media/two_robot_gripper/stills/ppo_*.png`
- 실패 장면 stills: `docs/training_media/two_robot_gripper/stills/untrained_*.png`

## 한계와 보고서에 적을 내용

이 환경은 “완전한 로봇 grasp를 RL이 처음부터 발견하는 연구급 문제”가 아니라, course-project용으로 빠르게 결과를 보여주기 위한 hybrid control 구조이다.

- Robot A의 초기 밀기는 scripted primitive이다.
- Robot B의 grasp는 fixed constraint primitive로 안정화했다.
- PPO는 Robot B의 관절 residual과 gripper close를 학습한다.
- target block 선택은 별도 고수준 decision이 아니라 curriculum target으로 고정/랜덤 선택된다.

이 구조의 장점은 발표에서 로봇이 직접 젠가를 밀고, 커스텀 gripper로 잡아 뽑고, 바닥에 떨어뜨리는 장면을 보여줄 수 있다는 점이다. 한계는 실제 tactile grasp, 블록 선택 전략, 여러 턴 장기 생존 전략은 후속 확장으로 남는다는 점이다.

## Refined precise-pusher update

사용자 피드백에 따라 Robot A가 긴 막대처럼 타워 전체를 치는 모습을 줄였다.

변경 내용:

- Robot A pusher tool을 짧은 tip 형태로 변경했다.
- pusher tip이 target block face에 맞도록 offset을 수정했다.
- Robot A가 target block만 약 5cm 노출시키도록 velocity-capped force를 조정했다.
- `pusher_neighbor_disturbance` metric을 추가해서 초기 pusher가 다른 블록을 얼마나 흔드는지 기록한다.
- Robot B reward에서 `neighbor_delta` penalty를 `8.0 -> 14.0`으로 강화했다.
- non-target gripper contact penalty도 `0.04 -> 0.20`으로 강화했다.

Fine-tuning 결과:

```text
old model, 30 seeds: completed=0.333, collapsed=0.667, neighbor_motion=0.182
fine-tuned model, 30 seeds: completed=0.467, collapsed=0.533, neighbor_motion=0.152
```

발표용 refined success rollout:

```text
GIF: docs/training_media/two_robot_gripper_precise/ppo/rollout_seed_237.gif
completed=True
collapsed=False
floor_dropped=True
neighbor_motion=0.013 m
pusher_neighbor_disturbance=0.003 m
```

## No-approach-collision update

추가 피드백: Robot A의 파란 막대가 내려오는 동안 target block을 먼저 건드려서, 조준 전에 블록이 저절로 밀려나오는 문제가 있었다.

수정 내용:

- Robot A pusher tip의 collision을 제거했다.
- 접근 중 막대는 visual-only로 보이지만 블록과 물리 충돌하지 않는다.
- 정렬 후에만 target block에 capped force를 적용해서 의도적으로 약 5cm 노출시킨다.
- 접근 중 accidental movement를 확인하기 위해 `pusher_approach_displacement`와 `pusher_approach_neighbor_disturbance`를 추가했다.

검증 결과(seed 0~11 reset diagnostic):

```text
max pusher_approach_displacement = 0.0 m
max pusher_approach_neighbor_disturbance ≈ 0.00002 m
mean intentional exposure ≈ 0.038 m
```

새 발표용 GIF:

```text
docs/training_media/two_robot_gripper_no_approach_collision/ppo/rollout_seed_237.gif
```

해당 성공 rollout:

```text
completed=True
collapsed=False
floor_dropped=True
pusher_approach_displacement=0.0 m
pusher_target_displacement=0.058 m
pusher_neighbor_disturbance≈0.00003 m
neighbor_motion=0.012 m
```

## Robot A safe-path / camera update

추가 피드백: Robot A 위치와 접근 경로 때문에, 파란 pusher가 내려오면서 젠가를 자동으로 치는 것처럼 보였다.

수정 내용:

- Robot A base를 대각선 sweep 위치에서 tower front 위치로 이동했다.
- Robot A를 target sampling 전에 high park pose로 배치했다.
- 접근 경로를 `safe hover -> outside lower -> horizontal approach`로 변경했다.
- 접근 중 pusher는 visual-only이고, 정렬 후에만 target block에 controlled force를 적용한다.
- Two-robot 환경 전용 카메라를 Robot B 쪽으로 옮겨 Robot A가 tower 앞을 가리지 않게 했다.

검증 결과:

```text
pusher_approach_displacement=0.0 m
pusher_approach_neighbor_disturbance≈0.00002 m
```

최종 발표용 GIF:

```text
docs/training_media/two_robot_gripper_final_camera/ppo/rollout_seed_237.gif
```
