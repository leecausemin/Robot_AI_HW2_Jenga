# Jenga 강화학습 로봇 과제 계획

## 1. 최종 목표

Franka Panda 로봇팔이 Jenga tower에서 규칙에 맞는 블록을 추출하고, 실제 Jenga 규칙처럼 그 블록을 tower의 맨 위에 다시 배치하는 강화학습 환경을 만든다.

과제 요구사항 기준 핵심은 두 가지다.

1. **환경**: Gymnasium + PyBullet + Panda robot 기반의 Jenga manipulation environment를 만든다.
2. **알고리즘**: PPO를 기본으로 학습하고, 추가 실험으로 Maskable PPO를 제시한다. 비교용으로 Random, Greedy, Q-Learning, DQN, A2C를 둘 수 있다.

## 2. 수업 PPT와의 연결

이 프로젝트는 `13_Capstone (1).pdf`의 구조를 따른다.

```text
Agent -> action -> Environment(World)
Environment = Robot + Task + Physics Engine
Robot = Franka Panda
Physics Engine = PyBullet
Interface = Gymnasium
```

또한 기말 PPT의 RL 흐름과 연결된다.

```text
MDP 정의
Value / Q function
Q-Learning
DQN
Policy Gradient
Actor-Critic
PPO
```

## 3. 메인 환경

메인 환경은 다음 파일에 구현되어 있다.

```text
src/jenga_rl/envs/stack_env.py
```

환경 이름:

```python
JengaPandaStackEnv
```

한 episode step은 Jenga의 한 턴을 의미한다.

```text
1. 최상단이 아닌 legal block 선택
2. Panda end-effector가 target block으로 이동
3. block 추출
4. 추출한 block을 top layer로 운반
5. top layer의 선택한 slot에 배치
6. tower lower support가 무너지지 않으면 성공
```

## 4. Jenga 규칙 제약

환경은 다음 규칙을 반영한다.

```text
최상단 블록 선택 금지
이미 제거한 블록 선택 금지
한 층의 마지막 지지 블록 선택 제한
이미 사용한 top placement slot 선택 금지
collapse 발생 시 episode 종료
```

agent가 사용할 수 있도록 다음 정보를 제공한다.

```python
info["legal_actions"]
info["action_mask"]
```

## 5. MDP 설계

### Observation

Capstone/panda-gym 스타일에 맞춰 robot state와 task state를 함께 포함한다.

```text
end-effector position
end-effector velocity
finger width
target block position
tower stability score
removed ratio
legal action ratio
top placement goal position
holding/placing phase indicator
```

### Action

현재는 discrete high-level action이다.

```python
action = extraction_action * 3 + placement_slot
extraction_action = source_level * 3 + source_slot
placement_slot = 0, 1, 2
```

즉 agent는 “어느 블록을 뺄지”와 “어디에 다시 놓을지”를 결정한다. IK와 motion primitive는 환경 내부에서 처리한다.

### Reward

```text
블록에 접근하면 작은 dense reward
블록 추출 성공 시 positive reward
top placement 성공 시 더 큰 positive reward
tower stability 유지 시 shaping reward
collapse 발생 시 큰 penalty
한 턴 완료 시 completion bonus
```

## 6. 알고리즘 계획

상세 내용은 `docs/ALGORITHM_PLAN.md`에 정리한다.

실험 순서:

1. Random baseline
2. Greedy rule-based baseline
3. Q-Learning / DQN / A2C 비교
4. **PPO 메인 학습**
5. **Maskable PPO 후순위 개선 실험**

Maskable PPO는 수업에서 이름 그대로 배운 알고리즘은 아니므로, “PPO 기반 action-mask 응용”으로 설명한다.

## 7. 검증 계획

환경 검증:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

렌더링 데모:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/demo_stack.py --steps 1 --out docs/jenga_panda_stack_after_turn.png
```

기대 결과:

```text
extracted=True
placed=True
collapsed=False
completed=True
```

## 8. 결과가 안 좋을 때 분석 포인트

- Jenga는 한 번의 선택이 미래 안정성에 영향을 주는 long-horizon problem이다.
- top placement까지 포함하면 extraction과 placement를 모두 성공해야 해서 reward가 sparse해진다.
- 일반 PPO는 illegal action을 sampling할 수 있어 sample efficiency가 낮아질 수 있다.
- Maskable PPO는 action mask로 이 문제를 완화하지만, PPO 응용이므로 추가 실험으로 다룬다.
- 실제 contact-rich manipulation은 PyBullet에서도 불안정할 수 있어, IK/motion primitive를 prior knowledge로 사용했다.


## 9. Active Probing 확장 계획

독창성을 높이기 위한 active probing 단계는 `JengaPandaProbeStackEnv`로 구현했고, 설계 배경은 `docs/PROBING_PLAN.md`에 정리했다. 핵심은 사람이 Jenga를 할 때처럼 로봇이 먼저 블록을 살짝 probe하고, 그 결과로 looseness/risk를 추정한 뒤 안전한 블록을 추출하는 것이다.

```text
probe -> infer risk -> extract -> top placement
```

이 확장은 단순 pick-and-place가 아니라 partial observability, active sensing, risk-aware decision making을 포함하므로 프로젝트 독창성을 크게 높인다.

## End-effector modeling update

The robot model was updated after identifying that the default Franka Panda parallel gripper is not ideal for Jenga. The current model uses a `jenga_probe` end-effector: a slim single-finger probe/pusher attached to the Panda end-effector and aligned with the selected block direction.

This supports the project concept better because Jenga requires narrow-gap probing and controlled pushing rather than generic two-finger pick-and-place.

Updated 10-level render:

```text
docs/training_media/robot_arm_10level/robot_attempt_10level_probe_tool.gif
```
