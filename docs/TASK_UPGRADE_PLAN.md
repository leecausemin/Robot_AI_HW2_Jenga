# Task upgrade plan: from block-selection to robot Jenga manipulation

## 문제점

현재 `JengaPandaEnv`는 Capstone 스타일의 Panda + PyBullet + Gymnasium 환경이지만, agent action이 아직 `어느 블록을 밀 것인가`에 가깝다. IK와 push trajectory는 환경 내부 rule로 처리하므로 과제 설명은 쉽지만, task complexity는 낮아 보일 수 있다.

## 개선 목표

Jenga 규칙과 robot manipulation을 모두 반영한 episodic task로 확장한다.

```text
1. legal block 찾기
2. end-effector가 target block에 접근하기
3. block을 tower 밖으로 안정적으로 추출하기
4. 추출한 block을 top layer 위에 올리기
5. tower가 무너지지 않으면 성공
```

## MDP 설계

### Observation

Capstone PPT의 panda-gym 관찰 구조에 맞춰 robot state + object state + goal을 포함한다.

- Robot state
  - end-effector position 3
  - end-effector velocity 3
  - finger width 1
- Object state
  - target block pose
  - target block velocity
  - tower center-of-mass/stability score
  - removed/held/placed stage
- Goal
  - desired extraction pose
  - desired top placement pose
- Rule state
  - legal action mask
  - locked top layer
  - per-level remaining blocks

### Action options

#### Option A: Hybrid discrete-continuous

```text
action = [block_id, approach_side, push_depth, push_speed]
```

장점: Jenga 의사결정과 조작 파라미터를 함께 학습한다.  
단점: SB3 기본 알고리즘 연결이 약간 복잡하다.

#### Option B: Continuous end-effector control

```text
action = [dx, dy, dz, gripper]
```

장점: panda-gym과 가장 유사하다. PPO/SAC/TD3에 적합하다.  
단점: 학습 난이도와 시간이 크게 증가한다.

#### Option C: Curriculum

1. Reach target block
2. Push/extract target block
3. Extract without collapse
4. Extract and place on top

장점: 과제 제출용으로 설명이 좋고 실패 원인 분석도 자연스럽다.

## Reward 설계

- Dense reward
  - end-effector와 target block 거리 감소
  - target block extraction distance 증가
  - tower stability 유지
  - top placement goal에 가까워짐
- Sparse reward
  - successful extraction
  - successful top placement
  - collapse penalty
  - illegal top-layer action penalty

## Jenga rule constraints

- Top layer locked: top level actions are illegal.
- Removed blocks cannot be selected again.
- A layer's final support block is locked for safety.
- Collapse terminates episode.
- Optional official rule extension: after extracting a block, place it on top before next turn.

## 추천 구현 순서

1. 현재 `JengaPandaEnv` 유지: Capstone-style baseline environment.
2. `JengaPandaExtractEnv` 추가: target block extraction만 continuous control로 학습.
3. `JengaPandaExtractAndStackEnv` 추가: extraction + top placement.
4. 알고리즘 비교:
   - Q-learning: discrete block-selection baseline only
   - DQN: discrete block-selection baseline
   - PPO: hybrid/discrete or continuous variant
   - SAC/TD3: continuous end-effector control variant
5. 결과 분석:
   - simple block-selection은 greedy가 강할 수 있음
   - continuous robot control은 exploration/sample efficiency 문제가 큼
   - sparse reward는 HER 또는 curriculum 없이는 잘 안 나올 수 있음

## 발표용 문장

단순히 블록 번호를 고르는 환경에서 시작했지만, 최종 목표는 Capstone PPT의 Panda-gym 구조처럼 robot state, object state, goal state를 포함하는 manipulation task이다. 특히 Jenga의 실제 규칙을 반영하여 최상단 블록은 action mask로 금지하고, agent는 접근 방향과 push 깊이 또는 end-effector displacement를 학습하도록 확장할 수 있다.
