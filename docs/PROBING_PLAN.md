# Active Probing Jenga Robot 계획

## 1. 최종 컨셉

### 프로젝트 제목 후보

**만져보고 판단하는 젠가 로봇: Active Probing 기반 강화학습**

영문 제목:

**Risk-Aware Jenga Manipulation with Active Probing using Reinforcement Learning**

## 2. 스토리

일반적인 robot manipulation task는 물체를 집어서 목표 위치에 놓는 pick-and-place가 많다. 하지만 Jenga는 단순한 pick-and-place가 아니다. 어떤 블록을 빼도 되는지 판단해야 하고, 잘못 선택하면 전체 tower가 무너진다.

사람은 Jenga를 할 때 보통 다음과 같이 행동한다.

```text
1. 아무 블록이나 바로 빼지 않는다.
2. 손가락으로 여러 블록을 살짝 밀어본다.
3. 잘 움직이는 블록은 안전하다고 판단한다.
4. 뻑뻑하거나 tower가 흔들리는 블록은 포기한다.
5. 안전한 블록을 빼서 맨 위에 올린다.
```

이 프로젝트는 이 사람의 전략을 로봇 강화학습 환경으로 만든다.

```text
Panda robot이 Jenga block을 probe한다.
probe 결과로 block의 looseness와 tower risk를 추정한다.
안전한 block을 선택해 추출한다.
추출한 block을 top layer에 다시 배치한다.
```

핵심은 단순히 block을 잘 빼는 것이 아니라, **불확실한 물리 상태를 능동적으로 탐색하고 위험을 줄이는 robot policy를 학습하는 것**이다.

## 3. 현재 환경에서의 발전 방향

현재 구현된 메인 환경:

```text
JengaPandaStackEnv
= Panda robot + PyBullet + Gymnasium
= select block -> extract -> top placement
```

업그레이드 목표:

```text
JengaPandaProbeStackEnv
= probe -> infer risk -> extract -> top placement
```

## 4. 독창성 평가

| 단계 | 환경 | 독창성 |
| --- | --- | ---: |
| 현재 | Panda가 block을 추출하고 top placement | 6.5 / 10 |
| 개선 1 | hidden looseness 추가 | 7.5 / 10 |
| 개선 2 | probe action 추가 | 8.5 / 10 |
| 개선 3 | probe history 기반 risk-aware extraction | 9 / 10 |
| 개선 4 | multi-turn strategy + uncertainty-aware policy | 9.5~10 / 10 |

10점에 가까워지려면 단순 제어가 아니라 다음 키워드가 들어가야 한다.

```text
partial observability
active sensing
risk-aware decision making
rule-constrained reinforcement learning
multi-stage robot manipulation
```

## 5. 환경 설계

### 5.1 Hidden block properties

각 block은 reset 시 보이지 않는 물리 속성을 가진다.

```python
looseness[level, slot]       # 잘 빠지는 정도
resistance[level, slot]      # probe/push에 대한 저항
support_risk[level, slot]    # 제거 시 tower 안정성 위험
```

agent는 이 값을 직접 보지 못한다.

대신 probe 결과를 통해 추정해야 한다.

### 5.2 Episode phase

환경은 multi-stage task로 구성한다.

```text
Phase 1: Probe
Phase 2: Select/Extract
Phase 3: Carry
Phase 4: Top placement
Phase 5: Stability check
```

### 5.3 Action space 후보

#### Option A: Discrete high-level action

과제 제출 안정성을 고려하면 가장 추천한다.

```python
action_type = probe or extract_place
probe_action = target_block
extract_place_action = target_block + placement_slot
```

하나의 Discrete action으로 encoding 가능하다.

```text
action 0 ~ N-1          : probe block i
action N ~ N+N*3-1      : extract block i and place at top slot j
```

장점:

- PPO/DQN/Q-learning 비교가 가능하다.
- Maskable PPO 적용이 쉽다.
- 수업 내용과 연결이 쉽다.

단점:

- raw robot control 학습은 아니다.

#### Option B: Dict or MultiDiscrete action

```python
{"mode": probe/extract, "block": i, "placement_slot": j}
```

장점:

- 의미적으로 깔끔하다.

단점:

- Stable-Baselines3 연결이 번거롭다.

#### Option C: Continuous end-effector control

```python
action = [dx, dy, dz, gripper]
```

장점:

- panda-gym과 가장 유사하다.
- SAC/TD3/HER와 연결 가능하다.

단점:

- 학습 시간이 길고 구현 난이도가 높다.
- 과제 제출 리스크가 크다.

### 최종 추천

```text
Option A: Discrete high-level action
```

이 방식이 과제 완성도, 수업 알고리즘 비교, 보고서 설명에 가장 좋다.

## 6. Observation 설계

현재 `JengaPandaStackEnv` observation에 probe 정보를 추가한다.

### Robot state

```text
end-effector position
end-effector velocity
finger width
```

### Object/task state

```text
target block position
tower stability score
removed ratio
legal action ratio
top placement goal position
current phase
```

### Probe history

각 block에 대해 최근 probe 결과를 저장한다.

```text
probe_count[level, slot]
last_probe_displacement[level, slot]
last_probe_resistance[level, slot]
last_stability_change[level, slot]
```

### Hidden vs observed

| 항목 | agent 관찰 가능 여부 |
| --- | --- |
| looseness | 직접 관찰 불가 |
| resistance | probe 후 추정 가능 |
| support risk | probe 후 stability change로 추정 가능 |
| tower stability | 관찰 가능 |
| legal action mask | 관찰 가능 |

## 7. Reward 설계

### Probe reward

Probe 자체에 큰 reward를 주면 agent가 계속 probe만 할 수 있다. 그래서 작은 정보 보상 또는 cost를 둔다.

```text
probe cost: -0.02
safe information gain: +0.05
risky probe causing instability: -0.3
```

### Extraction reward

```text
successful extraction: +1.0
extracting previously probed safe block: +0.3
extracting unprobed risky block: -0.2
collapse: -8.0
```

### Top placement reward

```text
successful top placement: +2.0
stable tower after placement: +3.0
full turn success bonus: +5.0
```

### Final reward intuition

좋은 policy는 다음 행동을 학습해야 한다.

```text
probe를 너무 많이 하지 않는다.
위험한 block은 피한다.
잘 움직이는 block을 선택한다.
뺀 block을 top layer에 안정적으로 놓는다.
```

## 8. Jenga rule constraints

유지할 규칙:

```text
top layer block cannot be extracted
already removed block cannot be selected
last support block in a layer is restricted
already used placement slot cannot be selected
collapse terminates episode
```

추가할 규칙:

```text
maximum probe count per episode
same block repeated probing penalty
must extract and place to complete a turn
```

## 9. 알고리즘 계획

### 기본 실험

1. Random
2. Greedy probing heuristic
3. Q-Learning
4. DQN
5. A2C
6. PPO

### 추가 실험

7. Maskable PPO (fallback)

### 왜 PPO 먼저인가?

PPO는 수업에서 배운 Policy Gradient / Actor-Critic 계열과 연결된다. 구현 안정성이 높고, high-level discrete action task에도 적용하기 쉽다.

### 왜 Maskable PPO를 추가로 쓰는가?

Jenga는 legal/illegal action이 명확하다. Maskable PPO는 action mask를 이용해 불가능한 행동을 제거하므로, PPO가 illegal action 때문에 잘 학습되지 않을 때 후순위 개선안으로 사용한다. 단, 수업에서 이름 그대로 배운 것은 아니므로 PPO의 응용 실험으로 제시한다.

## 10. 비교 실험 설계

### Experiment 1: No probing vs probing

| 환경 | 설명 |
| --- | --- |
| NoProbe | 현재 `JengaPandaStackEnv` |
| Probe | probe history가 포함된 `JengaPandaProbeStackEnv` |

가설:

```text
Probe 환경에서 PPO가 collapse rate를 더 낮출 것이다.
```

### Experiment 2: PPO vs Maskable PPO

| 알고리즘 | 설명 |
| --- | --- |
| PPO | 수업 기반 기본 알고리즘 |
| Maskable PPO | legal action mask 사용 |

가설:

```text
Maskable PPO가 illegal action rate를 낮추고 success rate를 높일 것이다.
```

### Experiment 3: Probe cost ablation

| 설정 | 예상 |
| --- | --- |
| probe cost 없음 | agent가 probe만 반복할 수 있음 |
| 작은 probe cost | 적절한 탐색 유도 |
| 큰 probe cost | probe를 거의 하지 않음 |

## 11. 평가 지표

```text
mean return
full-turn success rate
extraction success rate
top-placement success rate
collapse rate
illegal action rate
average probe count
safe block selection rate
sample efficiency
```

## 12. 결과가 잘 안 나올 때 분석 포인트

- Probe 정보가 reward와 충분히 연결되지 않으면 agent가 probe를 무시할 수 있다.
- Probe cost가 너무 작으면 probe만 반복할 수 있다.
- Probe cost가 너무 크면 exploration을 하지 않는다.
- Jenga는 long-horizon stability problem이라 credit assignment가 어렵다.
- PyBullet contact simulation은 미세한 마찰/힘 조절에서 불안정할 수 있다.
- PPO는 illegal action을 sampling할 수 있어 Maskable PPO보다 sample efficiency가 낮을 수 있다.

## 13. 구현 단계

### Step 1: Hidden looseness 추가

- reset마다 block별 looseness/resistance/risk 생성
- block dynamics의 friction/mass 또는 simulated probe response에 반영

### Step 2: Probe action 추가

- action space를 probe action + extract/place action으로 확장
- probe는 작은 motion primitive로 block을 살짝 밀고 결과를 기록

### Step 3: Probe observation 추가

- probe_count
- last_probe_displacement
- last_probe_resistance
- last_stability_change

### Step 4: Reward 수정

- probe cost
- safe information reward
- extraction/top-placement reward
- collapse penalty

### Step 5: PPO 학습 계획 작성

- PPO 기본 실험
- Maskable PPO 후순위 실험
- no-probe vs probe 비교

## 14. 최종 발표 스토리

### 제목

**만져보고 판단하는 젠가 로봇**

### 발표 흐름

1. 기존 pick-and-place task는 단순히 물체를 옮기는 문제이다.
2. Jenga는 어떤 블록이 안전한지 알 수 없는 불확실한 manipulation 문제이다.
3. 사람은 블록을 살짝 밀어보고 안전한 블록을 찾는다.
4. 본 프로젝트는 이 전략을 Panda robot + PyBullet + Gymnasium 환경으로 구현한다.
5. agent는 probe 결과를 바탕으로 위험을 줄이며 block을 추출하고 top layer에 배치한다.
6. PPO를 기본으로 학습하고, PPO가 illegal action/sample efficiency 문제로 충분히 학습되지 않을 때 Jenga rule constraint를 활용하는 Maskable PPO를 후순위로 비교한다.

### 한 문장 요약

> 이 프로젝트는 단순한 Jenga block selection이 아니라, 로봇이 직접 블록을 만져보고 위험을 판단한 뒤 안전한 블록을 추출하고 top layer에 배치하는 risk-aware robot manipulation task이다.
