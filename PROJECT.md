# Jenga Reinforcement Learning Robot Project

## 1. 프로젝트 개요

이 프로젝트는 **PyBullet 3D 물리 시뮬레이션 환경에서 두 대의 Panda 로봇이 협력하여 Jenga 블록을 안전하게 추출하는 강화학습 과제**이다.

기본 아이디어는 사람이 Jenga를 할 때 사용하는 전략을 로봇 태스크로 단순화한 것이다.

1. 먼저 한쪽 로봇이 목표 블록을 살짝 밀어서 밖으로 노출시킨다.
2. 반대쪽 로봇이 노출된 끝부분만 조심스럽게 잡는다.
3. 목표 블록을 타워에서 완전히 빼낸다.
4. 다른 블록은 최대한 건드리지 않고, 타워가 무너지지 않도록 한다.
5. 최종적으로 목표 블록을 바닥에 떨어뜨리면 성공으로 본다.

최신 메인 환경은 다음 파일에 구현되어 있다.

```text
src/jenga_rl/envs/two_robot_gripper_env.py
```

환경 이름은 다음과 같다.

```text
TwoRobotJengaGripperEnv
```

발표용 최종 GIF는 다음 경로에 있다.

```text
docs/training_media/two_robot_gripper_far_b_exposed_tip/ppo/rollout_seed_237.gif
```

---

## 2. 프로젝트 목표

### 2.1 강화학습 목표

이 프로젝트에서 학습시키는 핵심 목표는 **Robot B가 target block을 안정적으로 잡고 빼는 행동을 학습하는 것**이다.

Robot A는 학습 대상이 아니라, target block을 노출시키는 scripted helper로 사용한다. 즉, 전체 작업은 다음처럼 나뉜다.

| 구성 요소 | 역할 | 학습 여부 |
|---|---|---|
| Robot A | target block을 약간 밀어서 밖으로 노출 | 학습하지 않음, scripted primitive |
| Robot B | 노출된 target block을 잡고 당겨서 바닥에 떨어뜨림 | PPO로 학습 |
| Jenga tower | 물리 기반 시뮬레이션 대상 | 학습 대상 아님 |

### 2.2 성공 조건

episode가 성공하려면 다음 조건을 만족해야 한다.

```text
completed = ever_grasped and floor_dropped and target_displacement > 0.14 and not collapsed
```

즉,

1. Robot B가 target block을 한 번이라도 grasp해야 한다.
2. target block이 바닥 높이까지 떨어져야 한다.
3. target block이 충분히 추출되어야 한다.
4. 타워가 붕괴하면 안 된다.

---

## 3. Environment

### 3.1 환경 이름

```text
TwoRobotJengaGripperEnv
```

### 3.2 기반 라이브러리

이 환경은 다음 라이브러리 기반으로 구성되어 있다.

- `Gymnasium`: 강화학습 환경 인터페이스
- `PyBullet`: 3D 물리 시뮬레이션과 렌더링
- `stable-baselines3`: PPO 학습
- `NumPy`: 상태 계산, 보상 계산, 벡터 연산

### 3.3 환경 구성

환경은 다음 구성 요소를 가진다.

| 요소 | 설명 |
|---|---|
| Jenga tower | PyBullet box body로 구성된 3D Jenga tower |
| Robot A | Franka Panda robot, target block을 미는 pusher 역할 |
| Robot B | Franka Panda robot, custom Jenga gripper를 가진 학습 agent |
| Custom gripper | Jenga 블록을 잡기 위한 긴 빨간 jaw 형태의 gripper |
| Target block | 이번 episode에서 추출해야 하는 블록 |
| Non-target blocks | 건드리면 안 되는 나머지 블록 |

### 3.4 Tower 설정

현재 two-robot 환경의 기본 tower 설정은 다음과 같다.

```text
levels = 6
blocks_per_level = 3
max_removed_blocks = 1
```

즉, 6층짜리 Jenga tower에서 한 개의 target block을 추출하는 단일 추출 task이다.

블록마다 마찰 계수가 무작위로 설정된다.

```text
friction ~ Uniform(0.30, 1.20)
```

이렇게 해서 매 episode마다 블록의 움직임이 조금씩 달라진다. 실제 Jenga에서 어떤 블록은 잘 빠지고, 어떤 블록은 빡빡한 느낌을 단순화해서 반영한 것이다.

### 3.5 Target block 선택

현재 curriculum은 빠른 학습과 안정적인 발표 결과를 위해 단순화되어 있다.

- level 1의 edge block 중 하나를 target으로 선택한다.
- 가운데 블록은 제외한다.
- 너무 높은 층은 제외한다.

코드상으로는 다음 후보를 사용한다.

```text
level = 1
slot != 1
```

이렇게 한 이유는 이 프로젝트의 핵심이 “어떤 블록을 고를지”보다 **선택된 블록을 로봇팔로 조심스럽게 추출하는 제어 문제**이기 때문이다.

---

## 4. Agent

### 4.1 학습 agent

학습 agent는 **Robot B**이다.

Robot B는 Franka Panda 로봇팔이고, 일반 Panda gripper 대신 Jenga에 맞춘 custom gripper primitive를 사용한다.

Robot B의 역할은 다음과 같다.

1. target block의 노출된 끝부분에 접근한다.
2. gripper를 닫아 target block을 잡는다.
3. target block을 타워 밖으로 당긴다.
4. 충분히 추출한 후 아래로 내려 바닥에 떨어뜨린다.
5. non-target block을 최대한 움직이지 않게 한다.

### 4.2 Robot A

Robot A는 학습하지 않는다.

Robot A는 target block을 살짝 밀어 밖으로 노출시키는 scripted pusher이다. 발표 영상에서 파란색 pusher tip으로 보인다.

Robot A의 설계 의도는 다음과 같다.

- 사람이 손가락으로 Jenga 블록을 살짝 밀어 loose한지 확인하거나 끝부분을 노출시키는 행동을 모방한다.
- Robot B가 바로 묻혀 있는 블록 중심을 잡으려 하지 않도록 target block의 바깥 끝부분을 만들어준다.
- Robot A가 조준 전에 블록을 치는 문제를 피하기 위해 pusher tip은 visual-only로 두고, 정렬 후에만 target block에 작은 capped force를 적용한다.

### 4.3 Custom gripper

기본 Panda gripper는 Jenga 블록을 잡기에는 폭과 형상이 적합하지 않다. 따라서 이 프로젝트에서는 Jenga 전용 custom gripper를 사용한다.

custom gripper 구성:

- 검은색 palm
- 빨간색 긴 jaw 2개
- target block의 노출된 끝부분을 잡는 구조

중요한 제약:

- Robot B의 일반 Panda link collision은 비활성화한다.
- task contact는 custom gripper primitive와 fixed grasp constraint로 처리한다.
- grasp 전에는 gripper contact point가 tower 안쪽으로 들어가지 못하도록 outside-limit constraint를 둔다.

이 제약을 둔 이유는 발표 영상에서 Robot B의 몸체가 tower를 뚫고 지나가거나, target block이 아닌 다른 블록을 건드리는 것처럼 보이는 문제를 막기 위해서이다.

---

## 5. Action Space

### 5.1 Action 정의

Robot B의 action space는 8차원 continuous action이다.

```python
action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
```

각 action의 의미는 다음과 같다.

| Action index | 의미 |
|---|---|
| 0~6 | Robot B의 7개 Panda arm joint residual command |
| 7 | custom gripper close command |

즉, PPO policy가 직접 내는 action은 다음 두 부분으로 나뉜다.

```text
action = [joint_residual_0, ..., joint_residual_6, gripper_close]
```

### 5.2 Joint residual control

Robot B의 joint action은 완전한 raw torque control이 아니라, **motion primitive + residual learning** 구조이다.

환경 내부에는 기본적으로 target block을 당기는 방향의 prior action이 있다.

```text
command = jacobian_prior_action + residual_scale * policy_residual
```

여기서 PPO는 전체 동작을 처음부터 새로 배우는 것이 아니라, 기본 motion primitive에서 얼마나 보정할지를 학습한다.

이 구조를 사용한 이유는 다음과 같다.

- 순수 joint control을 처음부터 학습하면 시간이 너무 오래 걸린다.
- 짧은 과제 시간 안에 안정적인 결과를 얻기 어렵다.
- 사람이 설계한 기본 동작 위에 RL이 보정량을 학습하면 빠르게 수렴한다.

### 5.3 Gripper close command

마지막 action 값은 gripper를 닫는 정도에 영향을 준다.

```text
gripper_closed = clip(gripper_closed + 0.20 * action[7], 0.0, 1.0)
```

`gripper_closed`가 0.55 이상이고, gripper가 target grasp site와 충분히 가까우면 fixed constraint를 만들어 target block을 grasp한 것으로 처리한다.

---

## 6. Observation Space

### 6.1 Observation 크기

observation space는 36차원 continuous vector이다.

```python
observation_space = Box(low=-20.0, high=20.0, shape=(36,), dtype=np.float32)
```

### 6.2 Observation 구성

observation은 다음 요소들로 구성된다.

| 구간 | 차원 | 설명 |
|---|---:|---|
| Robot B joint position `q` | 7 | Panda arm 7개 joint angle |
| Robot B joint velocity `dq` | 7 | Panda arm 7개 joint velocity |
| End-effector position | 3 | Robot B end-effector의 3D 위치 |
| Target block position | 3 | target block의 3D 위치 |
| Grasp-site relative vector | 3 | `target_grasp_site - gripper_center` |
| Pull direction | 3 | target block을 당겨야 하는 방향 |
| Target displacement | 1 | target block이 추출 방향으로 이동한 거리 |
| Target height | 1 | target block의 현재 z 높이 |
| Grasped flag | 1 | 현재 grasp 여부 |
| Gripper closed value | 1 | gripper가 닫힌 정도 |
| Floor dropped flag | 1 | target block이 바닥에 떨어졌는지 여부 |
| Neighbor motion | 1 | non-target block 중 가장 많이 움직인 정도 |
| Stability score | 1 | tower 안정도 점수 |
| Non-target gripper contacts | 1 | target이 아닌 블록과 gripper 접촉 수 |
| Level normalized | 1 | target block level 정규화 값 |
| Slot normalized | 1 | target block slot 정규화 값 |

총합:

```text
7 + 7 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 36
```

### 6.3 중요한 observation

발표에서 강조할 만한 observation은 다음이다.

#### 1. `target_grasp_site - gripper_center`

Robot B가 target block의 중심이 아니라, Robot A가 노출시킨 바깥쪽 끝부분을 기준으로 접근하게 한다.

이전에는 gripper가 block center를 목표로 해서 tower 안쪽으로 파고드는 문제가 있었다. 현재는 exposed-tip grasp site를 사용해서 이 문제를 줄였다.

#### 2. `neighbor_motion`

다른 블록이 얼마나 움직였는지를 나타낸다. Jenga task에서는 target block만 빼야 하고, 다른 블록이 움직이면 위험하다. 따라서 이 값은 reward penalty와 성공 여부 판단에 중요한 역할을 한다.

#### 3. `target_displacement`

target block이 얼마나 추출되었는지를 나타낸다. PPO는 이 값이 증가하도록 보상을 받는다.

---

## 7. Reward Function

### 7.1 Reward 설계 목적

reward는 다음 행동을 유도하도록 설계되어 있다.

1. target block을 추출 방향으로 많이 움직인다.
2. target block을 잡는다.
3. target block을 바닥까지 떨어뜨린다.
4. 다른 블록은 움직이지 않는다.
5. 타워를 무너뜨리지 않는다.
6. 너무 큰 action을 쓰지 않는다.

### 7.2 Reward 항목

코드상 reward는 다음 항목으로 구성된다.

```python
reward = 0.0
reward += 7.0 * progress
reward += 0.25 if grasped else -0.025 * min(5.0, gripper_target_distance)
reward += 0.15 * gripper_closed if not grasped else 0.0
reward -= 14.0 * neighbor_delta
reward -= 0.20 * non_target_gripper_contacts
reward -= 0.012 * mean(action[:7]^2)
if neighbor > 0.08:
    reward -= 2.0 * (neighbor - 0.08)
reward += 0.25 * (stability_after - stability_before)
if completed:
    reward += 16.0
if collapsed:
    reward -= 12.0
```

### 7.3 Reward 항목 해석

| Reward 항목 | 의미 |
|---|---|
| `+7.0 * progress` | target block이 추출 방향으로 이동하면 보상 |
| `+0.25 if grasped` | target block을 잡으면 보상 |
| `-0.025 * distance` | grasp 전에는 gripper가 target grasp site에 가까워지도록 유도 |
| `+0.15 * gripper_closed` | grasp 전에는 gripper를 닫도록 유도 |
| `-14.0 * neighbor_delta` | 다른 블록이 새로 움직이면 큰 penalty |
| `-0.20 * non_target_contacts` | target이 아닌 블록과 접촉하면 penalty |
| `-0.012 * action^2` | 너무 거친 joint action을 억제 |
| `-2.0 * (neighbor - 0.08)` | non-target motion이 일정 이상이면 추가 penalty |
| `+0.25 * stability improvement` | tower 안정도 유지 유도 |
| `+16.0 completed` | 성공 시 큰 terminal reward |
| `-12.0 collapsed` | 붕괴 시 큰 penalty |

### 7.4 Reward 설계 의도

단순히 target block을 빨리 빼는 것만 보상하면 로봇이 tower를 거칠게 치면서 블록을 빼려 할 수 있다. 그래서 이 프로젝트에서는 다음 균형을 맞추려고 했다.

```text
빠른 추출 보상 + 안전성 penalty + 성공 terminal reward
```

즉, PPO는 target block을 빼야 하지만, 동시에 다른 블록을 많이 흔들면 손해를 보도록 설계되어 있다.

---

## 8. Policy

### 8.1 사용한 Policy

학습에는 stable-baselines3의 PPO `MlpPolicy`를 사용했다.

```python
PPO("MlpPolicy", env, ...)
```

`MlpPolicy`는 observation vector를 입력으로 받아 continuous action vector를 출력하는 neural network policy이다.

입력:

```text
36-dimensional observation
```

출력:

```text
8-dimensional continuous action
```

### 8.2 Policy가 학습하는 것

policy는 다음을 학습한다.

- Robot B의 7개 arm joint residual
- gripper를 언제 얼마나 닫을지
- target block을 당길 때 어느 정도 joint 보정을 줄지
- target을 충분히 뺀 뒤 바닥으로 떨어뜨리는 동작

중요한 점은 policy가 모든 동작을 zero부터 학습하는 것이 아니라, environment가 제공하는 Jacobian prior 위에서 residual을 학습한다는 것이다.

발표에서는 다음처럼 설명하면 좋다.

> 저수준 motion primitive는 사람이 설계했고, PPO는 그 위에서 target block을 안전하게 추출하기 위한 보정 제어와 gripper closing timing을 학습했다.

---

## 9. Algorithm

### 9.1 사용 알고리즘: PPO

이 프로젝트의 메인 알고리즘은 **PPO(Proximal Policy Optimization)** 이다.

PPO를 선택한 이유는 다음과 같다.

1. continuous action space에 적합하다.
2. 로봇 joint residual control처럼 연속적인 제어 문제에 적용하기 쉽다.
3. stable-baselines3에서 안정적으로 구현되어 있다.
4. DQN/Q-learning보다 continuous action에 자연스럽다.
5. 짧은 시간 안에 어느 정도 안정적인 policy를 얻기 좋다.

### 9.2 왜 이 환경에서 PPO가 적합한가?

이 환경은 단순한 grid world나 discrete action 문제가 아니라, **로봇팔의 연속 제어 문제**이다. Agent인 Robot B는 매 step마다 다음과 같은 continuous action을 출력한다.

```text
7D Panda joint residual + 1D gripper close command
```

즉, 행동이 `왼쪽/오른쪽/위/아래`처럼 몇 개의 선택지로 나뉘는 것이 아니라, 각 joint에 대해 `-1.0 ~ 1.0` 사이의 실수값을 출력해야 한다. PPO는 이런 continuous control 문제에서 Gaussian policy를 사용해 연속적인 action distribution을 학습할 수 있기 때문에 이 환경과 잘 맞는다.

또한 Jenga 환경은 reward가 sparse하면서도 불안정하다. 예를 들어 target block을 조금씩 잘 빼면 보상이 증가하지만, 다른 블록을 흔들거나 tower가 무너지면 큰 penalty를 받는다. 따라서 policy가 한 번에 너무 크게 바뀌면 학습이 불안정해질 수 있다. PPO는 policy update의 크기를 clipping objective로 제한한다. 이 덕분에 로봇이 갑자기 너무 거친 동작을 학습하는 것을 줄이고, 비교적 안정적으로 개선할 수 있다.

이 프로젝트에서 PPO가 특히 적합했던 이유는 다음과 같이 정리할 수 있다.

| 환경 특성 | PPO가 적합한 이유 |
|---|---|
| Continuous action | joint residual과 gripper close가 실수값 action이므로 PPO의 continuous policy와 잘 맞음 |
| 물리 시뮬레이션 | PyBullet에서 작은 action 차이도 결과가 달라지므로 안정적인 policy update가 중요함 |
| 안전성 중요 | tower collapse penalty가 크기 때문에 급격한 policy 변화보다 보수적인 개선이 유리함 |
| Reward shaping 존재 | progress, grasp, neighbor penalty 등 여러 reward term을 on-policy 방식으로 안정적으로 반영 가능 |
| 짧은 학습 시간 | SAC/TD3처럼 많은 replay tuning을 하기보다, SB3 PPO로 빠르게 baseline 확보 가능 |
| 발표용 안정성 | 학습 결과와 rollout을 재현하기 쉽고 구현이 단순함 |

### 9.3 다른 알고리즘 대신 PPO를 사용한 이유

#### 9.3.1 Q-learning을 사용하지 않은 이유

Q-learning은 기본적으로 discrete state/action table을 전제로 한다. 하지만 이 프로젝트의 observation은 36차원 continuous vector이고, action도 8차원 continuous vector이다.

Q-learning을 사용하려면 다음을 해야 한다.

- joint angle을 구간으로 나누기
- gripper 위치를 구간으로 나누기
- action도 여러 discrete action으로 나누기

하지만 이렇게 하면 state-action 공간이 폭발적으로 커진다. 예를 들어 각 action dimension을 단순히 5개 값으로만 나누어도, 8차원 action은 다음 개수가 된다.

```text
5^8 = 390,625 actions
```

따라서 Q-learning은 이 로봇팔 제어 문제에 적합하지 않다.

#### 9.3.2 DQN을 사용하지 않은 이유

DQN은 neural network를 사용하지만, 기본적으로 discrete action을 출력하는 알고리즘이다. 이 프로젝트의 action은 Robot B의 7개 joint residual과 gripper close command로 구성된 continuous action이다.

DQN을 쓰려면 continuous action을 강제로 discrete action set으로 바꿔야 한다. 하지만 그렇게 하면 다음 문제가 생긴다.

- joint control이 거칠어진다.
- gripper가 부드럽게 접근하기 어렵다.
- action 조합 수가 너무 커진다.
- target block을 섬세하게 잡는 동작이 어려워진다.

Jenga task는 다른 블록을 건드리지 않는 섬세한 제어가 중요하므로, DQN보다 PPO가 더 적합하다.

#### 9.3.3 Maskable PPO를 메인으로 쓰지 않은 이유

Maskable PPO는 invalid action을 mask로 제거할 수 있는 discrete action 문제에 유용하다. 예를 들어 “어떤 Jenga block을 선택할 것인가?” 같은 high-level decision에는 Maskable PPO가 잘 맞을 수 있다.

하지만 현재 최종 환경의 핵심은 block selection이 아니라 **Robot B의 continuous joint residual control**이다. 즉, agent가 선택해야 하는 것은 block index가 아니라 다음과 같은 실수 action이다.

```text
joint residual = [-0.3, 0.1, ..., 0.2]
gripper close = 0.8
```

따라서 action masking보다 continuous policy learning이 더 중요하다. 그래서 최종 메인 알고리즘은 PPO로 결정했다.

발표에서는 다음처럼 설명할 수 있다.

> Maskable PPO는 어떤 블록을 선택할지 결정하는 discrete task에는 적합하지만, 본 프로젝트의 최종 task는 로봇 관절 residual을 제어하는 continuous control 문제이므로 PPO가 더 적합하다.

#### 9.3.4 A2C를 사용하지 않은 이유

A2C도 continuous action에 적용할 수 있고 PPO와 비슷한 actor-critic 계열이다. 하지만 PPO는 A2C보다 policy update를 더 안정적으로 제한한다.

Jenga 환경에서는 작은 제어 실수로도 tower가 무너질 수 있다. 따라서 안정적인 update가 중요하다. PPO는 clipping objective를 통해 이전 policy에서 너무 멀리 벗어나지 않도록 학습한다. 이 점 때문에 A2C보다 PPO가 더 안전한 선택이다.

#### 9.3.5 SAC/TD3를 사용하지 않은 이유

SAC와 TD3는 continuous control에서 강력한 off-policy 알고리즘이다. 하지만 이 프로젝트에서는 다음 이유로 PPO를 우선 선택했다.

- 구현과 튜닝이 PPO보다 복잡하다.
- replay buffer 기반 학습에서 reward shaping과 termination 조건 튜닝이 더 민감할 수 있다.
- 과제 목표는 최고 성능의 로봇 제어 연구가 아니라, 환경 설계와 학습 결과를 명확히 보여주는 것이다.
- PPO는 stable-baselines3에서 기본 예제가 많고, 학습/평가/발표용 rollout을 만들기 쉽다.

즉, SAC/TD3가 이론적으로 더 sample-efficient할 가능성은 있지만, 과제 상황에서는 PPO가 안정성, 구현 난이도, 설명 가능성 측면에서 더 적합했다.

### 9.4 PPO 선택 논리 요약

이 프로젝트에서 PPO를 선택한 논리는 다음 한 문장으로 정리할 수 있다.

> 이 환경은 36차원 연속 observation을 보고 8차원 연속 action으로 Robot B의 joint residual과 gripper close를 제어해야 하는 물리 기반 로봇 제어 문제이므로, discrete 알고리즘보다 continuous policy gradient 계열이 적합하고, 그중 PPO는 clipping objective 덕분에 안정적으로 학습할 수 있어 최종 알고리즘으로 선택했다.

알고리즘 비교를 표로 정리하면 다음과 같다.

| 알고리즘 | 장점 | 이 프로젝트에서의 한계 | 최종 판단 |
|---|---|---|---|
| Q-learning | 구현이 단순, 이론 설명 쉬움 | continuous state/action에 부적합, table 크기 폭발 | 부적합 |
| DQN | neural network 사용 가능 | discrete action 전용, joint control을 discretize해야 함 | 부적합 |
| Maskable PPO | invalid discrete action 제거 가능 | block selection에는 적합하지만 joint residual control에는 부적합 | 보조 아이디어 |
| A2C | continuous action 가능 | PPO보다 update 안정성이 낮을 수 있음 | 가능하지만 PPO 선호 |
| SAC/TD3 | continuous control 성능 강함 | 튜닝 복잡, 과제 시간 대비 부담 큼 | future work |
| PPO | continuous action 가능, 안정적 update, 구현 쉬움 | sample efficiency가 SAC보다 낮을 수 있음 | 최종 선택 |

### 9.5 PPO 학습 설정

학습 스크립트는 다음 파일이다.

```text
scripts/train_two_robot_gripper_ppo.py
```

주요 PPO hyperparameter는 다음과 같다.

| Parameter | Value |
|---|---:|
| policy | `MlpPolicy` |
| n_steps | 128 |
| batch_size | 64 |
| n_epochs | 5 |
| gamma | 0.97 |
| learning_rate | 3e-4 |
| ent_coef | 0.01 |
| max_episode_steps | 90 |
| default timesteps | 8000 |

### 9.6 Early stopping

학습 중 일정 timestep마다 evaluation을 수행한다.

```text
eval_every = 1000
evaluation episodes = 5
early_stop_completion = 0.8
```

평가 성공률이 threshold 이상이면 조기 종료할 수 있다.

### 9.7 사용한 모델

발표용으로 사용한 모델은 다음 경로의 PPO checkpoint이다.

```text
runs/two_robot_gripper_precise_pusher_ft3k/model.zip
```

이 모델은 two-robot gripper task에서 target block을 잡고 추출하는 데 사용되었다.

---

## 10. Episode 진행 과정

하나의 episode는 다음 순서로 진행된다.

### Step 1. Reset

환경을 초기화한다.

- Jenga tower 생성
- Robot A, Robot B 생성
- block friction randomization
- target block 선택

### Step 2. Robot A scripted exposure

Robot A가 target block 앞까지 안전하게 접근한다.

접근 경로:

```text
safe hover -> outside lower -> horizontal approach
```

이후 target block에 작은 force를 적용해 약 5cm 정도 밖으로 노출시킨다.

중요한 검증 값:

```text
pusher_approach_displacement = 0.0
```

이는 Robot A가 접근하는 도중에 target block을 실수로 밀지 않았다는 뜻이다.

### Step 3. Robot B exposed-tip approach

Robot B가 tower 뒤쪽에서 접근한다.

Robot B는 target block 중심이 아니라, Robot A가 노출시킨 바깥쪽 끝부분을 목표로 한다.

접근 경로:

```text
safe hover -> outside lower -> outside pre-grasp -> target grasp
```

### Step 4. PPO control

PPO policy가 매 step마다 8차원 action을 출력한다.

- 7D joint residual
- 1D gripper close command

Robot B는 target을 잡고 당긴 뒤, 바닥 방향으로 내려 target block을 떨어뜨린다.

### Step 5. Termination

다음 중 하나가 발생하면 episode가 끝난다.

- `completed = True`
- `collapsed = True`
- `step_count >= max_episode_steps`

---

## 11. Jenga 규칙 반영

이 프로젝트는 일반 Jenga 규칙을 완전히 구현한 것은 아니지만, 과제 목표에 맞게 일부 규칙을 환경 제약으로 반영했다.

### 반영한 규칙

| Jenga 규칙/전략 | 환경 반영 방식 |
|---|---|
| 한 번에 한 블록만 조작 | `max_removed_blocks = 1` |
| target block만 조작 | target block과 non-target block을 구분 |
| 다른 블록을 건드리면 위험 | `neighbor_motion`, `non_target_contacts` penalty |
| tower가 무너지면 실패 | `collapsed=True` 시 episode 종료 및 penalty |
| 너무 위쪽 블록은 제외 | 현재 curriculum에서 lower/middle target 사용 |
| block마다 빡빡함이 다름 | block friction randomization |
| 노출된 끝부분을 잡음 | exposed-tip grasp site 사용 |

### 단순화한 부분

| 실제 Jenga | 현재 프로젝트 단순화 |
|---|---|
| 여러 턴 동안 블록을 빼고 위에 쌓음 | 현재는 한 episode에서 한 블록 추출 |
| 사람은 모든 블록 후보를 탐색 | 현재 target block은 환경이 선택 |
| 실제 gripper contact physics | fixed constraint 기반 grasp primitive 사용 |
| 완전한 관절/토크 제어 | Jacobian prior + PPO residual control |

---

## 12. Rendering / Visualization

### 12.1 렌더링 방식

렌더링은 PyBullet camera image를 사용한다.

환경은 `render_mode="rgb_array"`를 지원한다.

```python
env = TwoRobotJengaGripperEnv(render_mode="rgb_array")
frame = env.render()
```

### 12.2 발표용 카메라

발표용 카메라는 Robot B 쪽에서 tower를 바라보도록 설정되어 있다.

```text
cameraEyePosition = [1.20, 1.05, 0.78]
cameraTargetPosition = [0.56, 0.00, 0.27]
```

이 카메라 각도는 다음을 잘 보여주기 위해 선택했다.

- Robot A가 target block을 노출시키는 모습
- Robot B가 뒤쪽에서 gripper로 접근하는 모습
- target block이 빠지는 모습
- tower가 무너지지 않는 모습

### 12.3 발표용 media

최신 발표용 GIF:

```text
docs/training_media/two_robot_gripper_far_b_exposed_tip/ppo/rollout_seed_237.gif
```

이 GIF의 핵심 결과:

```text
completed=True
collapsed=False
floor_dropped=True
gripper_inside_penetration=0.0
non_target_gripper_contacts=0
neighbor_motion≈0.027 m
step_count=14
```

---

## 13. 실험 결과 요약

최신 성공 rollout의 결과는 다음과 같다.

| Metric | Value |
|---|---:|
| completed | True |
| collapsed | False |
| floor_dropped | True |
| extracted | True |
| target_displacement | 약 0.222 m |
| target_height | 약 0.041 m |
| neighbor_motion | 약 0.027 m |
| non_target_gripper_contacts | 0 |
| gripper_inside_penetration | 0.0 |
| step_count | 14 |

이 결과는 다음을 의미한다.

- target block을 성공적으로 잡았다.
- target block을 충분히 추출했다.
- target block이 바닥으로 떨어졌다.
- tower는 붕괴하지 않았다.
- gripper가 grasp 전에 tower 안쪽으로 파고들지 않았다.
- target이 아닌 블록과 gripper 접촉은 기록되지 않았다.

---

## 14. Baseline / 비교 관점

이 프로젝트는 개발 과정에서 여러 단계의 baseline을 거쳤다.

### 14.1 초기 접근: single robot push

처음에는 한 대의 로봇이 블록을 밀어서 완전히 빼는 방식을 고려했다.

문제점:

- 밀기만으로는 블록이 완전히 바닥으로 떨어지지 않는 경우가 있었다.
- 로봇팔이 target이 아닌 블록을 건드리는 것처럼 보였다.
- target block을 안정적으로 잡는 동작이 부족했다.

### 14.2 개선 접근: two robot gripper

이후 두 대의 로봇을 사용했다.

- Robot A: pusher
- Robot B: custom gripper

장점:

- 사람이 Jenga를 할 때처럼 한쪽에서 살짝 밀고, 반대쪽에서 잡아 빼는 구조가 된다.
- target block을 바닥으로 떨어뜨리는 성공 조건을 더 명확히 만들 수 있다.
- 발표 영상에서 task story가 분명해진다.

### 14.3 알고리즘 비교 관련

현재 메인 결과는 PPO를 사용한다. Maskable PPO는 discrete action mask가 필요한 경우에 유용하지만, 이 프로젝트의 최종 action은 continuous joint residual control이므로 PPO가 더 자연스럽다.

발표에서는 다음처럼 말할 수 있다.

> 초기에는 어떤 블록을 선택할지까지 포함한 discrete decision problem에서는 Maskable PPO를 고려했다. 하지만 최종 프로젝트는 로봇 관절 residual과 gripper close를 제어하는 continuous control 문제로 바뀌었기 때문에, 최종 알고리즘은 PPO를 사용했다.

---

## 15. 독창성 포인트

이 프로젝트의 독창성은 단순히 “Jenga를 하는 로봇”이 아니라, **Jenga에 맞는 역할 분담과 gripper 제약을 설계했다는 점**이다.

### 15.1 Two-robot cooperation

한 로봇이 모든 것을 하는 대신, 두 로봇이 역할을 나눈다.

```text
Robot A: expose target block
Robot B: grasp and extract target block
```

이 구조는 실제 사람이 Jenga 블록을 빼낼 때 한쪽에서 밀고, 튀어나온 부분을 잡아 빼는 전략과 비슷하다.

### 15.2 Exposed-tip grasp

Robot B는 block center를 잡으려 하지 않는다. 대신 Robot A가 노출시킨 바깥쪽 끝부분만 잡는다.

이 설계는 다음 문제를 해결한다.

- gripper가 tower를 통과해 보이는 문제
- target이 아닌 블록을 건드리는 문제
- Robot B 몸체가 tower에 너무 가까워 보이는 문제

### 15.3 Safety-aware reward

reward는 단순히 빠르게 빼는 것만 보상하지 않는다.

- target progress 보상
- non-target movement penalty
- collapse penalty
- action magnitude penalty
- stability score 보상

따라서 agent는 “빨리 빼기”와 “안전하게 빼기” 사이의 균형을 학습한다.

---

## 16. 한계점

이 프로젝트는 과제 시간 안에 안정적인 결과와 발표 가능한 영상을 만드는 것을 목표로 했기 때문에 몇 가지 단순화가 있다.

### 16.1 완전한 grasp physics는 아님

custom gripper는 visible primitive와 fixed constraint 기반 grasp를 사용한다. 즉, 실제 gripper contact force만으로 블록을 잡는 완전한 물리 grasp는 아니다.

이 선택을 한 이유:

- 실제 grasp physics까지 학습하면 난이도가 크게 올라간다.
- Panda 기본 gripper는 Jenga 블록을 잡기에 적합하지 않다.
- 짧은 시간 안에 학습 가능한 환경을 만들기 위해 task primitive를 사용했다.

### 16.2 Robot A는 학습하지 않음

Robot A는 scripted helper이다. 따라서 전체 시스템 중 학습 대상은 Robot B이다.

향후 확장한다면 Robot A의 probe force나 pusher trajectory도 학습시킬 수 있다.

### 16.3 한 번의 추출 task

현재는 여러 턴 동안 Jenga를 오래 하는 multi-turn game이 아니라, 한 episode에서 하나의 target block을 추출하는 task이다.

향후 확장:

- 여러 block을 순차적으로 추출
- 제거한 block을 tower 위에 다시 쌓기
- tower stability를 장기적으로 유지하는 multi-turn reward

### 16.4 Target block selection은 단순화

현재 target block은 lower level edge block에서 선택된다. 즉, agent가 “어떤 블록을 선택할지”를 학습하지는 않는다.

향후 확장:

- high-level policy가 block 선택
- low-level policy가 grasp/extraction control
- hierarchical RL 구조 적용

---

## 17. 향후 개선 방향

발표에서 future work로 넣을 수 있는 내용은 다음과 같다.

### 17.1 Multi-turn Jenga

현재는 한 블록만 빼지만, 실제 Jenga처럼 여러 턴을 진행하도록 확장할 수 있다.

추가할 요소:

- removed block count 증가
- tower top placement task
- collapse risk 누적
- 최대한 오래 버티는 survival reward

### 17.2 Block selection policy

현재는 target block이 정해져 있다. 다음 단계에서는 agent가 직접 어떤 블록을 뺄지 선택하게 만들 수 있다.

가능한 구조:

```text
High-level policy: block 선택
Low-level policy: 선택된 block 추출
```

### 17.3 Probe-based sensitivity estimation

각 블록의 friction이 다르기 때문에, Robot A가 블록을 살짝 probe해서 어느 블록이 loose한지 추정하는 기능을 넣을 수 있다.

가능한 observation:

- probe displacement
- contact response
- estimated looseness
- block friction category

### 17.4 More realistic gripper physics

현재는 fixed constraint를 사용하지만, 향후에는 실제 collision shape과 contact force를 사용한 grasp로 확장할 수 있다.

---

## 18. 발표용 설명 문장

발표에서 사용할 수 있는 설명은 다음과 같다.

> 이 프로젝트는 PyBullet 기반 3D Jenga 환경에서 두 대의 Panda 로봇이 협력하여 target block을 추출하는 강화학습 프로젝트입니다. Robot A는 scripted pusher로 target block을 살짝 노출시키고, Robot B는 PPO로 학습된 policy를 이용해 노출된 끝부분을 잡고 블록을 바닥으로 떨어뜨립니다. Observation은 Robot B의 joint state, target block 위치, gripper와 target grasp site 사이의 상대 위치, tower stability, neighbor block motion 등을 포함합니다. Reward는 target block 추출 progress를 보상하면서, 다른 블록 움직임과 tower collapse를 강하게 penalty로 주어 안전한 추출을 유도합니다.

짧은 버전:

> 목표는 Jenga tower를 무너뜨리지 않고 target block 하나를 안전하게 빼는 것입니다. Robot A가 블록을 살짝 노출시키고, PPO로 학습된 Robot B가 custom gripper로 노출된 끝부분만 잡아 추출합니다.

---

## 19. 실행 방법

### 19.1 테스트

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yumin/jenga/.venv/bin/python -m pytest -q
```

### 19.2 PPO 학습

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/train_two_robot_gripper_ppo.py \
  --timesteps 8000 \
  --out runs/two_robot_gripper_ppo \
  --levels 6 \
  --max-episode-steps 90
```

### 19.3 GIF 기록

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_two_robot_gripper_rollout.py \
  --policy ppo \
  --model runs/two_robot_gripper_precise_pusher_ft3k/model.zip \
  --seed 237 \
  --out-dir docs/training_media/two_robot_gripper_far_b_exposed_tip \
  --levels 6 \
  --max-episode-steps 90
```

---

## 20. 핵심 요약

| 항목 | 내용 |
|---|---|
| Project | Two-robot Jenga block extraction |
| Environment | `TwoRobotJengaGripperEnv` |
| Simulator | PyBullet |
| RL library | stable-baselines3 |
| Agent | Robot B, Panda + custom gripper |
| Helper | Robot A, scripted pusher |
| Algorithm | PPO |
| Policy | MlpPolicy |
| Observation | 36D continuous vector |
| Action | 8D continuous vector |
| Reward | extraction progress + grasp + safety penalty + success bonus |
| Success | target block extracted and dropped, tower not collapsed |
| Main media | `docs/training_media/two_robot_gripper_far_b_exposed_tip/ppo/rollout_seed_237.gif` |

