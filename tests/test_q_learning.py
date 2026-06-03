from __future__ import annotations

from jenga_rl.training.q_learning import QLearningConfig, evaluate_q_learning, train_q_learning


def test_q_learning_runs_short_training_loop() -> None:
    learner, returns = train_q_learning(QLearningConfig(episodes=3, seed=3))
    assert len(returns) == 3
    assert len(learner.q) > 0
    metrics = evaluate_q_learning(learner, episodes=2, seed=100)
    assert 0.0 <= metrics["collapse_rate"] <= 1.0
