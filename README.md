# Jenga RL Robot Project

강화학습 과제용 **Jenga block-removal environment + algorithm comparison** 프로젝트입니다.
현재 구현은 빠르게 학습/비교 가능한 Gymnasium 스타일의 추상 물리 환경입니다. PyBullet/MuJoCo 로봇 시뮬레이터는 보고서 확장안으로 남겨두고, 과제 핵심인 환경 설계와 알고리즘 비교를 먼저 안정적으로 통과하도록 구성했습니다.

## 핵심 아이디어

- **환경**: 18층 Jenga tower에서 로봇이 `(level, slot)` 블록을 선택해 제거합니다.
- **관측**: tower occupancy, 제거 개수, 안정성 margin, 남은 legal action 비율.
- **행동**: `Discrete(54)` = `level * 3 + slot`.
- **보상**: 안정적인 제거 성공 보상, collapse penalty, 여러 블록 제거 완료 bonus.
- **비교 대상**: random, greedy stability, tabular Q-learning, PPO, A2C, DQN.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[train,dev]
```

## 환경 검증

```bash
python scripts/check_env.py
pytest
```

## 알고리즘 비교

빠른 baseline만:

```bash
python scripts/compare.py --episodes 200
```

SB3 알고리즘까지 짧게 학습:

```bash
python scripts/compare.py --episodes 50 --q-episodes 1000 --timesteps 20000
```

개별 학습:

```bash
python scripts/train.py --algo ppo --timesteps 50000
python scripts/train.py --algo a2c --timesteps 50000
python scripts/train.py --algo dqn --timesteps 50000
```

## 코드 구조

```text
src/jenga_rl/
  envs/                  # Gymnasium-compatible Jenga environment
  baselines.py           # random / greedy baseline policies
  training/sb3_runner.py # PPO, A2C, DQN construction + training
  evaluation/sb3_eval.py # trained-agent evaluation
scripts/                 # check, train, compare entrypoints
tests/                   # environment regression tests
docs/                    # project plan and report template
```

## 보고서에서 강조할 점

1. 환경을 일반 Gymnasium API로 만들어서 알고리즘을 쉽게 바꿀 수 있음.
2. Jenga 안정성을 support polygon과 center-of-mass margin으로 모델링함.
3. 단순 random보다 greedy/RL이 더 긴 episode와 낮은 collapse rate를 내는지 비교함.
4. 결과가 나쁘면 sparse reward, long-horizon credit assignment, 추상 물리 모델의 한계를 분석함.

## 수업 내용 연결

수업 PPT와의 연결은 `docs/COURSE_ALIGNMENT.md`에 정리했습니다. 핵심은 MDP 정의 → Q-Learning 구현 → DQN/PPO/A2C 비교 순서입니다.

## 3D 시각화

현재 RL 환경은 빠른 학습을 위한 상태 기반 MDP이지만, 같은 tower state를 3D 이미지로 확인할 수 있습니다.

```bash
pip install -e .[train]
python scripts/render_3d.py --steps 4 --out docs/jenga_tower_3d.png
```

주의: 이 3D 렌더러는 발표/보고서용 시각화입니다. 실제 rigid-body contact physics까지 포함하려면 PyBullet 또는 MuJoCo 환경으로 확장하면 됩니다.

## 다음 확장

현재 task가 단순한 문제를 해결하기 위한 확장안은 `docs/UPGRADE_PLAN.md`에 정리했습니다. 핵심 방향은 block selection에서 끝내지 않고, Panda robot이 block extraction과 top placement까지 수행하는 manipulation task로 확장하는 것입니다.

## Main upgraded environment: JengaPandaStackEnv

The main environment is now `JengaPandaStackEnv`, which performs a full Jenga turn: select a legal non-top block, extract it with the Panda robot, carry it, and place it on the top layer. See `docs/STACK_ENV.md`.

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/demo_stack.py --steps 1 --out docs/jenga_panda_stack_after_turn.png
```

## Originality upgrade: JengaPandaProbeStackEnv

The originality-focused environment is `JengaPandaProbeStackEnv`: the robot first probes a candidate block, observes looseness/risk signals, then extracts the selected block and places it on top. See `docs/PROBE_STACK_ENV.md`.

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/demo_probe_stack.py --out docs/probe_stack_turn.png
```
