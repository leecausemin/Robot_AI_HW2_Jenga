# T3 Jenga Panda RL 학습 정리

## 1. 목표

이 프로젝트의 목표는 **로봇팔이 젠가 타워를 최대한 오래 유지하면서 블록을 안전하게 제거하는 것**이다.
현재 실험 목표는 다음과 같다.

```text
타워: 10층 Jenga tower
목표: 붕괴 없이 합법적인 블록 3개 제거
규칙: 최상단 층 선택 금지, 이미 제거한 블록 재선택 금지, 한 층의 마지막 지지 블록 제거 금지
전략: 바로 밀지 않고, 먼저 살짝 probe 해서 느슨한 블록을 찾은 뒤 push-extract
```

즉, 단순히 블록 하나를 고르는 문제가 아니라 **probe → 물리 반응 관찰 → 안전한 블록 추출**을 학습하는 강화학습 문제로 만들었다.

## 2. 환경(Environment)

메인 환경은 `JengaPandaProbeStackEnv` 하나로 정리했다.

```text
파일: src/jenga_rl/envs/probe_stack_env.py
기반: Gymnasium + PyBullet
렌더링/로봇: Franka Panda robot arm + blue Jenga probe tool
타워: PyBullet rigid-body Jenga blocks, 10층
과제: active probing + pure push extraction
```

### 왜 top placement를 제거했나?

기존 젠가처럼 위에 다시 쌓는 task도 생각했지만, Panda 기본 그리퍼는 현재 블록 폭을 안정적으로 집기 어렵다. 그래서 가짜 teleport grasp를 넣는 대신, 이번 버전은 더 정직하게 **로봇이 실제로 밀어서 빼는 pure push extraction**으로 구성했다.

## 3. 에이전트(Agent)

에이전트는 `Maskable PPO`를 사용했다.

```text
알고리즘: Maskable PPO
라이브러리: stable-baselines3 + sb3-contrib
정책: MLP policy
학습 단위: discrete high-level action
```

로봇이 관절 토크를 처음부터 학습하는 것이 아니라, 강화학습 에이전트는 고수준 의사결정만 한다.

```text
RL agent가 결정하는 것:
- 어느 블록을 probe 할지
- 어느 블록을 push-extract 할지

환경/로봇 primitive가 처리하는 것:
- Panda IK 이동
- probe tool 위치 이동
- 물리 기반 push force 적용
- tower stability/collapse 판정
```

이렇게 나눈 이유는 관절 제어까지 직접 RL로 학습하면 수백만 timestep이 필요할 수 있어서, 과제 범위에서는 고수준 의사결정 학습이 더 적절하기 때문이다.

## 4. Action / Observation / Reward

### Action

타워 블록 수를 `N`이라고 하면 action은 다음처럼 구성된다.

```text
0 <= action < N      : 해당 블록을 probe
N <= action < 2N     : 해당 블록을 push-extract
```

환경은 `action_mask`를 제공해서 젠가 규칙 위반 action을 막는다.

```text
- 최상단 층 블록 선택 금지
- 제거된 블록 선택 금지
- 한 층의 마지막 지지 블록 제거 금지
- probe 횟수 제한
```

### Observation

관측에는 다음 정보가 포함된다.

```text
- 타워 안정도
- 제거된 블록 수
- 각 블록의 probe 여부/횟수
- probe 시 실제 변위와 저항
- 현재까지 관측된 좋은 후보 블록
```

마찰계수는 직접 보여주지 않는다. 에이전트는 probe 결과를 통해 간접적으로 느슨한 블록을 추정해야 한다.

### Reward

20k 학습에서는 active probing을 더 잘 배우도록 reward를 조정했다.

```text
probe: 작은 비용 + 실제로 잘 움직인 블록이면 signal bonus
probed extraction 성공: 큰 보상
blind extraction: penalty
stuck extraction: penalty
collapse: 큰 penalty
3개 제거 완료: completion bonus
```

핵심은 **무작정 미는 정책보다, 먼저 probe 하고 안전한 블록을 고르는 정책이 유리하도록 만든 것**이다.

## 5. 학습 방식

이번 최종 실험은 다음 조건으로 수행했다.

```bash
PYTHONPATH=src .venv/bin/python scripts/train_probe_stack.py \
  --algo maskable_ppo \
  --timesteps 20000 \
  --seed 11 \
  --out runs/t3_maskable_probe20k \
  --levels 10 \
  --max-removed-blocks 3 \
  --primitive-step-scale 0.2
```

학습 요약:

```text
model: runs/t3_maskable_probe20k/maskable_ppo/model.zip
timesteps: 20,000
elapsed: 약 3,598초, 약 60분
training mode: fast_dynamics=True
render/demo mode: full_robot, Panda IK + blue probe push
```

학습 중 평균 reward는 초반 음수에서 시작해 후반에는 약 `11~12` 수준까지 올라갔다. 5k 실험보다 확실히 안정적인 정책을 얻었다.

## 6. 평가 결과

20 episode 평가 결과:

| Policy | Mean return | Avg survived turns / 3 | Completion rate | Collapse rate | Avg probe count | Avg steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | -7.834 | 1.90 | 0.25 | 0.05 | 13.65 | 27.00 |
| Greedy probe | -1.071 | 1.85 | 0.25 | 0.05 | 2.60 | 24.35 |
| Maskable PPO 20k | 12.350 | 3.00 | 1.00 | 0.00 | 6.00 | 12.50 |

해석:

```text
Maskable PPO는 20/20 episode에서 목표인 3개 제거를 모두 성공했다.
collapse rate도 0%였다.
평균 probe count가 6.0이라, 무작정 미는 것이 아니라 probe를 활용하는 정책을 학습했다.
```

## 7. 발표용 자료 경로

학습된 정책 full robot GIF:

```text
docs/training_media/t3_full_robot_probe20k/maskable_ppo/rollout.gif
```

최종 full-robot render는 Panda base를 타워 뒤쪽으로 약간 이동시키고, 긴 blue probe tip만 선택 블록에 닿도록 조정했다. 학습 정책 GIF seed는 501이며, 3/3 제거 성공 및 collapse 없음이다. 최종 렌더링에서는 probe tip이 현재 블록 면을 따라가도록 하여 블록을 통과해 보이지 않게 했다.

비학습 정책 full robot GIF:

```text
docs/training_media/t3_full_robot_probe20k/untrained/rollout.gif
```

비학습 비교 GIF seed는 502이며, 안전하지 않은 선택으로 tower collapse가 발생한다.

대표 이미지:

```text
docs/training_media/t3_full_robot_probe20k/maskable_ppo_start.png
docs/training_media/t3_full_robot_probe20k/maskable_ppo_mid.png
docs/training_media/t3_full_robot_probe20k/maskable_ppo_final.png
docs/training_media/t3_full_robot_probe20k/untrained_start.png
docs/training_media/t3_full_robot_probe20k/untrained_mid.png
docs/training_media/t3_full_robot_probe20k/untrained_final.png
```

평가 지표 CSV:

```text
docs/training_media/metrics/t3_probe20k_after_training.csv
```

## 8. 보고서에 쓸 수 있는 핵심 문장

> 본 프로젝트는 로봇이 젠가 블록을 무작위로 제거하는 것이 아니라, 실제 물리 시뮬레이션에서 probe를 통해 블록의 느슨함을 관찰하고, Maskable PPO를 이용해 젠가 규칙을 만족하는 안전한 push-extraction 정책을 학습하도록 설계했다.

> 로봇의 저수준 관절 제어는 IK 기반 motion primitive로 처리하고, 강화학습 에이전트는 어느 블록을 probe/extract할지 결정하는 고수준 전략을 학습한다. 이를 통해 제한된 timestep 안에서도 10층 젠가 타워에서 3개 블록 제거를 안정적으로 달성했다.
