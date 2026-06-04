# Joint-Control Jenga Push Experiment

## 목적

이 폴더(`/home/yumin/jenga_joint_push`)는 기존 high-level 블록 선택 과제와 분리한 실험이다. 목표는 **어떤 블록을 뺄지 고르는 것**이 아니라, reset 때 랜덤으로 지정된 젠가 블록 하나를 **Panda 로봇의 7개 관절 delta action으로 안전하게 밀어 빼는 것**이다.

- 환경: `SingleBlockJointPushEnv`
- 알고리즘: PPO (`stable-baselines3`)
- 에이전트 action: 7차원 연속 action = Panda arm 7개 관절의 작은 위치 변화량
- 성공 조건: target block displacement > `0.10 m` 이고 tower collapse가 없을 것
- 실패 조건: tower collapse 또는 제한 step 초과

## 환경 설계

기존 Jenga/Panda 렌더링과 PyBullet tower/robot 구성을 복사해서 사용했다. 다만 학습 목표를 빠르게 만들기 위해 새 환경에서는 다음처럼 단순화했다.

1. **타깃 블록은 reset 때 랜덤 지정**
   - agent는 블록 선택을 학습하지 않는다.
   - 최상단 locked layer는 선택하지 않는다.

2. **관절 제어만 학습**
   - action은 `(7,)` Box action이다.
   - 각 action은 Panda arm joint position target에 작은 delta로 적용된다.

3. **빠른 커리큘럼 시작 위치**
   - 팔은 처음부터 target block 앞쪽에 거의 닿은 상태로 배치된다.
   - 이유: 5시간 안에 결과를 만들기 위해 reaching 전체를 학습시키지 않고, “어떤 관절 움직임으로 천천히 밀어야 안전하게 빠지는가”에 집중했다.

4. **블록 민감도 / 난이도**
   - reset 때 각 블록의 lateral friction을 랜덤화한다.
   - friction이 높은 블록은 더 빡빡하고, 밀 때 주변 블록을 더 긁어서 disturbance가 커지도록 했다.
   - agent가 friction 값을 직접 보지는 않는다. 대신 접촉 여부, target displacement, neighbor motion을 관측하면서 closed-loop로 조절해야 한다.

## 관측값(observation)

총 29차원:

- joint position 7
- joint velocity 7
- probe tip position 3
- probe tip에서 target face까지의 vector 3
- target push direction 3
- target displacement 1
- neighbor motion 1
- tower stability 1
- contact flag 1
- target level/slot normalized 2

## 보상(reward)

- target block이 앞으로 조금씩 움직이면 보상
- 접촉을 유지하면 보상
- 주변 블록 motion이 커지면 penalty
- action이 너무 크면 작은 penalty
- collapse면 큰 penalty
- target extraction 성공 + collapse 없음이면 큰 성공 보상

즉, “무작정 세게 치기”보다 **접촉을 유지하면서 천천히 밀어 빼는 관절 움직임**이 유리하도록 만들었다.

## 학습 결과

가장 좋은 짧은 run:

- run: `runs/joint_push_ppo_fast_v3`
- best model: `runs/joint_push_ppo_fast_v3/best_model.zip`
- requested timesteps: 10,000
- early stop: 1,000 timesteps에서 eval completion rate 0.8 도달
- 최종 별도 seed 평가: completion rate 0.6

중요: 1k timestep의 매우 짧은 실험이므로 완전히 안정적인 로봇 정책은 아니다. 그래도 같은 seed 300에서 아래처럼 학습 전/후 차이가 나온다.

| Policy | Seed | Target | Result | Target displacement | Neighbor motion | Steps |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| PPO learned | 300 | `(4, 0)` | 성공 | 0.1018 m | 0.0238 m | 52 |
| Untrained random | 300 | `(4, 0)` | 실패 | 0.0511 m | 0.0258 m | 60 |

## 발표용 media

- 학습된 PPO rollout GIF: `docs/training_media/joint_push/ppo/rollout.gif`
- 학습 전 random rollout GIF: `docs/training_media/joint_push/untrained/rollout.gif`
- 각 GIF의 frame PNG:  
  - `docs/training_media/joint_push/ppo/frames/`
  - `docs/training_media/joint_push/untrained/frames/`
- rollout 상세 로그:  
  - `docs/training_media/joint_push/ppo/summary.json`
  - `docs/training_media/joint_push/untrained/summary.json`

## 실행 명령

학습:

```bash
cd /home/yumin/jenga_joint_push
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/train_joint_push.py \
  --timesteps 10000 \
  --seed 43 \
  --out runs/joint_push_ppo_fast_v3 \
  --levels 6 \
  --max-episode-steps 60 \
  --eval-every 1000 \
  --early-stop-completion 0.8
```

렌더링/GIF:

```bash
PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_joint_push.py \
  --policy ppo \
  --model runs/joint_push_ppo_fast_v3/best_model.zip \
  --seed 300 \
  --out-dir docs/training_media/joint_push \
  --levels 6 \
  --max-episode-steps 60 \
  --record-every 3

PYTHONPATH=src /home/yumin/jenga/.venv/bin/python scripts/record_joint_push.py \
  --policy untrained \
  --seed 300 \
  --out-dir docs/training_media/joint_push \
  --levels 6 \
  --max-episode-steps 60 \
  --record-every 3
```

## 한계와 다음 개선

- 현재는 reaching을 처음부터 학습하지 않는다. 로봇은 target block 앞에서 시작한다.
- 완전한 torque-level 학습이 아니라 joint position delta control이다.
- 1k~8k timestep의 짧은 실험이라 seed에 따라 실패가 남아 있다.
- 더 좋은 결과를 원하면:
  1. 50k~200k timestep 학습
  2. target level curriculum: 쉬운 층부터 어려운 층으로 확장
  3. SAC/TD3 같은 continuous-control 알고리즘 비교
  4. separate probing phase 추가: 아주 약하게 먼저 건드려 friction을 추정한 뒤 push 속도를 조절
