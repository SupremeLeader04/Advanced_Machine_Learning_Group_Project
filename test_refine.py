"""Checks for the Section 2.4 code (refine.py and rlagent/agentE.py).

    pytest test_refine.py          (or: python test_refine.py)

Runs in about a minute. It only uses the group's environment, default
opponent and features; it does not change any files or results.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "rlagent"))

import features                                            # noqa: E402
import refine                                              # noqa: E402
from agentE import _state_features, policy                 # noqa: E402
from default_opponent import default_opponent_policy       # noqa: E402
from environment import sim_game                           # noqa: E402

THETA = np.load(Path(__file__).resolve().parent / "rlagent" / "agentE_theta.npy")


def test_weights_file_is_valid():
    """One row of 300 weights for each of the 10 periods, all finite."""
    assert THETA.shape == (10, 300)
    assert np.isfinite(THETA).all()


@pytest.mark.parametrize("budget", [0.0, 1e-9, 0.5, 20.0, 100.0, 1e4, 1e8, 1e12])
@pytest.mark.parametrize("t", [0, 5, 9])
def test_agent_respects_budget(budget, t):
    """delta >= 0, delta^2 + |x| <= B and finite actions, also for extreme budgets."""
    rng = np.random.default_rng(int(budget) % 1000 + t)
    for _ in range(50):
        state = (t, budget, rng.uniform(0, 1e4), rng.uniform(0, 400), rng.uniform(0, 400),
                 rng.normal(0, 1.5), rng.uniform(0, 1))
        delta, x = policy(*state)
        assert np.isfinite(delta) and np.isfinite(x)
        assert delta >= 0
        assert delta**2 + abs(x) <= budget


def test_effective_speed_cap():
    """The agent never pushes its effective speed above MAX_EFFECTIVE_SPEED."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        speed = rng.uniform(0, 400)
        delta, _ = policy(3, 10 ** rng.uniform(4, 12), 100.0, speed, 5.0, 1.0, 0.9)
        assert delta == 0 or speed + delta <= refine.MAX_EFFECTIVE_SPEED + 1e-9


def test_agent_features_match_group_features():
    """The copy of the 50 state indicators inside agentE.py equals features.state_features."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        s_self = rng.uniform(0, 15)
        s_opp = s_self if rng.uniform() < 0.1 else rng.uniform(0, 15)        # include ties
        state = [int(rng.integers(0, 10)), rng.choice([0.0, 20.0, 100.0, rng.uniform(0, 400)]),
                 rng.uniform(0, 300), s_self, s_opp,
                 rng.choice([0.0, 0.5, rng.normal(0, 1.5)]), rng.choice([0.0, 0.5, 1.0, rng.uniform(0, 1)])]
        assert np.array_equal(_state_features(*state), np.asarray(features.state_features(state), float))


def test_tournament_agent_matches_experiment_policy():
    """agentE.py (tournament file) and refine.py (experiments) give the same game outcomes."""
    n = 300
    np.random.seed(123)
    out = np.array([sim_game(policy_1=lambda s: list(policy(*s)),
                             policy_2=default_opponent_policy)[-1, 1:3] for _ in range(n)])
    b1, b2 = refine.rollouts(refine.greedy_batch(THETA), default_opponent_policy, n, seed=123)
    # identical up to floating-point dust (e.g. a budget of 1e-15 versus 2e-15)
    assert np.allclose(out[:, 0], b1, rtol=1e-9, atol=1e-9)
    assert np.allclose(out[:, 1], b2, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("opponent", list(refine.standard_pool()))
def test_games_run_in_group_environment(opponent):
    """Full games in the group's sim_game against every opponent type finish with valid budgets."""
    b1, b2 = refine.rollouts(refine.greedy_batch(THETA), refine.standard_pool()[opponent], 20, seed=7)
    assert np.isfinite(b1).all() and (b1 >= 0).all() and (b2 >= 0).all()


def test_training_is_reproducible():
    """A tiny finite-horizon LSPI run with a fixed seed gives identical weights twice."""
    cfg = dict(n_rounds=1, games_per_round=60, eval_games=20, seed=5, verbose=False)
    th1, _ = refine.train_backward(**cfg)
    th2, _ = refine.train_backward(**cfg)
    assert np.array_equal(th1, th2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
