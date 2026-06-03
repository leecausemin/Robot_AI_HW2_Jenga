# 보고서 템플릿

## 1. 문제 정의

본 프로젝트는 Franka Panda 로봇팔이 Jenga tower에서 legal block을 추출하고, 실제 Jenga 규칙처럼 추출한 block을 top layer에 다시 배치하는 강화학습 환경을 설계한다.

## 2. 수업 내용과 연결

- MDP: state, action, reward, transition, termination 정의
- Q-Learning / DQN: discrete action baseline
- Policy Gradient / Actor-Critic: PPO, A2C 비교
- Capstone: Panda robot, PyBullet, Gymnasium custom environment, robot manipulation task

## 3. 환경 모델링

### Environment

```text
JengaPandaStackEnv
Franka Panda + PyBullet + Gymnasium
```

### State / Observation

- end-effector position and velocity
- finger width
- target block position
- tower stability score
- removed ratio
- legal action ratio
- top placement goal position
- holding/placing phase indicator

### Action

```python
action = extraction_action * 3 + placement_slot
extraction_action = source_level * 3 + source_slot
placement_slot = 0, 1, 2
```

### Reward

- extraction success reward
- top placement success reward
- stability shaping reward
- collapse penalty
- completion bonus

### Termination

- tower collapse
- full turn success
- legal action 소진
- maximum step 초과

## 4. Jenga 규칙 반영

- 최상단 블록 선택 금지
- 이미 제거한 블록 선택 금지
- 한 층의 마지막 지지 블록 선택 제한
- 이미 사용한 placement slot 선택 금지
- action mask 제공

## 5. 알고리즘

| 구분 | 알고리즘 | 설명 |
| --- | --- | --- |
| Baseline | Random | 무작위 legal action 선택 |
| Baseline | Greedy | stability 기반 휴리스틱 |
| 수업 기반 | Q-Learning | tabular TD control |
| 수업 기반 | DQN | value-based deep RL |
| 수업 기반 | A2C | actor-critic baseline |
| 메인 | PPO | 수업 기반 policy optimization |
| 추가 | Maskable PPO | PPO에 action mask를 적용한 개선 실험 |

## 6. 실험 설정

- episodes:
- timesteps:
- seed:
- metric:
  - mean return
  - success rate
  - extraction success rate
  - top placement success rate
  - collapse rate
  - illegal action rate

## 7. 결과표

| Agent | Mean Return | Success Rate | Placement Rate | Collapse Rate | Illegal Action Rate | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random |  |  |  |  |  |  |
| Greedy |  |  |  |  |  |  |
| Q-Learning |  |  |  |  |  |  |
| DQN |  |  |  |  |  |  |
| A2C |  |  |  |  |  |  |
| PPO |  |  |  |  |  |  |
| Maskable PPO |  |  |  |  |  |  |

## 8. 결과 분석

PPO가 잘 나오는 경우:

- policy gradient가 high-level Jenga move selection에 안정적으로 적응했다.
- dense reward shaping이 extraction과 placement를 유도했다.

PPO가 잘 안 나오는 경우:

- legal action이 많고 invalid action도 있어 sample efficiency가 낮다.
- top placement까지 포함되어 reward가 sparse하다.
- contact-rich manipulation은 physics simulation에서 불안정하다.

Maskable PPO가 개선되는 경우:

- Jenga rule 기반 illegal action을 사전에 제거해 exploration 효율이 좋아진다.

## 9. 결론

본 프로젝트는 수업에서 배운 Gymnasium custom environment와 PPO를 기반으로 Panda robot Jenga manipulation task를 구성했다. PPO 결과가 충분하지 않을 경우, 후순위로 Jenga 규칙에 따른 action mask를 활용하는 Maskable PPO를 비교하여 rule-constrained robot manipulation에서 action masking의 효과를 분석한다.
