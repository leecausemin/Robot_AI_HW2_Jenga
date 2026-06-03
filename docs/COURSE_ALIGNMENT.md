# 수업 PPT 기반 정렬표

처음 만든 초안은 공개 GitHub 구조와 일반 RL 관행을 참고한 것이고, 수업 PPT를 직접 읽어서 만든 것은 아니었다. 이후 `/mnt/c/Users/user/Desktop/충남대학교/4_1/로봇AI/피피티`의 자료 범위에 맞춰 아래처럼 보강했다.

## 중간 범위와 연결

| PPT | 프로젝트 반영 |
| --- | --- |
| ROS2 Programming / Topic and Service / Action and Parameter | 실제 ROS2 노드는 아직 구현하지 않고, 보고서 확장안으로 로봇 제어 노드/Action server 구조를 제안 |
| Transformations | 실제 로봇팔 확장 시 tower frame, end-effector frame, camera frame을 TF로 연결하는 설계에 반영 가능 |
| Robot Modeling / ROS2 Control | PyBullet/MuJoCo 또는 URDF 기반 확장안에 반영 가능 |

## 기말 범위와 연결

| PPT | 코드/보고서 반영 |
| --- | --- |
| 2_MDP | `JengaTowerEnv`: state, action, reward, transition, termination 정의 |
| 3_Value Function & Bellman Equation | Q-value 기반 baseline/학습 설명에 반영 |
| 5_Monte Carlo Learning | episode return과 evaluation metric으로 반영 |
| 6_Temporal Difference Learning | Q-learning TD update 구현 |
| 7_Programming on Q-Learning | `src/jenga_rl/training/q_learning.py`: ε-greedy + Q-table 구현 |
| 10_Function Approximation | tabular state aggregation 설명, DQN 확장과 연결 |
| 11_Deep Q-Network | `scripts/compare.py --timesteps ...` 실행 시 DQN 비교 가능 |
| 12_Policy Gradient | PPO 비교 가능 |
| 13_Actor Critic Algorithms | A2C 비교 가능 |
| 14_Challenges and Recent Works | sparse reward, exploration, sim-to-real gap 분석에 반영 |

## 교수님께 설명하기 좋은 문장

이 프로젝트는 먼저 Jenga 문제를 MDP로 정의하고, 수업에서 다룬 tabular Q-learning을 기본 알고리즘으로 구현했다. 이후 function approximation 관점에서 DQN으로 확장하고, policy-based/actor-critic 계열 비교를 위해 PPO와 A2C를 추가했다. 실제 로봇 제어는 ROS2/URDF/TF/Control로 확장할 수 있지만, 본 과제에서는 강화학습 환경과 알고리즘 비교에 초점을 맞췄다.
