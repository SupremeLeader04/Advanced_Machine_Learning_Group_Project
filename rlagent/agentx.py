"""RL agent for the HFT arms-race tournament (Advanced Machine Learning, EFDS 2026-27).

Policy learned with finite-horizon LSPI (batch off-policy control with a
wealth-scaled linear Q-function; see refine.py in the main submission).
The agent does NOT learn during play and uses only the current state.

Usage:
    from agentx import policy
    delta, x = policy(t, B_self, B_opp, S_self, S_opp, g, R)

Only depends on numpy. Weights are loaded from agentx_theta.npy in this folder
(relative to this file, so it runs from any working directory).
"""

from math import erf, exp, sqrt, pi
from pathlib import Path

import numpy as np

_THETA = np.load(Path(__file__).resolve().parent / "agentx_theta.npy")   # (10, 50*6)

SIGMA = 1.5
T = 10
SPEED_FRACS = np.array([0, .0025, .005, .01, .02, .03, .05, .075, .1, .15,
                        .2, .3, .4, .5, .7, 1.0])
TRADE_FRACS = np.linspace(0, 1, 11)
U_GRID, V_GRID = [a.ravel() for a in np.meshgrid(SPEED_FRACS, TRADE_FRACS, indexing="ij")]
_Z = (np.arange(20) + 0.5) / 20
N_Z, N_H = 50, 6


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _npdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def _trade_gain(g, R, f):
    """E[max(0, 1 + f*sign(g)*Delta)] - 1 given the signal (per unit of wealth)."""
    mu = 1 + f * abs(g)
    s = max(f * SIGMA * sqrt(max(1 - R * R, 0.0)), 1e-9)
    return mu * _ncdf(mu / s) + s * _npdf(mu / s) - 1


def _onehot(i, k):
    v = np.zeros(k)
    v[i] = 1
    return v


def _state_features(t, B_self, B_opp, S_self, S_opp, g, R):
    """The group's 50 binary state indicators (same bins as features.py)."""
    d = S_self - S_opp
    sp = 0 if d < -5 else 1 if d < -3 else 2 if d < 0 else 3 if d == 0 else \
        4 if d < 3 else 5 if d < 5 else 6
    edges_B = [20, 40, 60, 80, 100, 120, 150]
    return np.concatenate([
        _onehot(int(t) // 2, 5),
        _onehot(int(np.searchsorted(edges_B, B_self, "right")), 8),
        _onehot(int(np.searchsorted(edges_B, B_opp, "right")), 8),
        _onehot(sp, 7),
        _onehot(int(np.searchsorted(np.arange(1, 10) / 10, R, "right")), 10),
        _onehot(0 if g < 0 else 1 if g == 0 else 2, 3),
        _onehot(int(np.searchsorted([.5, 1, 1.5, 2, 2.5, 3, 4, 10], abs(g), "right")), 9),
    ])


def _p_win(delta, S_self, S_opp, opp_scale):
    z = 2 * (S_self + delta[:, None] - S_opp - opp_scale * _Z[None, :])
    return (1 / (1 + np.exp(-np.clip(z, -50, 50)))).mean(axis=1)


def _action_terms(B_self, B_opp, S_self, S_opp, g, R):
    delta = np.sqrt(U_GRID * B_self)
    kept = 1 - U_GRID
    gains = np.array([_trade_gain(g, R, f) for f in TRADE_FRACS])
    gv = gains[np.round(V_GRID * 10).astype(int)]
    p_def = _p_win(delta, S_self, S_opp, sqrt(B_opp))
    p_pas = _p_win(delta, S_self, S_opp, 0.0)
    return np.stack([kept, kept * p_def * gv, kept * p_pas * gv,
                     kept * p_def, kept * p_pas, kept * V_GRID * p_def], axis=1)


def policy(t, B_self, B_opp, S_self, S_opp, g, R):
    """Return (delta, x): speed investment and trading position for this period."""
    t, B_self, B_opp = int(t), float(B_self), float(B_opp)
    S_self, S_opp, g, R = float(S_self), float(S_opp), float(g), float(R)
    if B_self <= 0:
        return 0.0, 0.0
    t_idx = min(max(t, 0), T - 1)
    z = _state_features(t_idx, B_self, max(B_opp, 0.0), max(S_self, 0.0),
                        max(S_opp, 0.0), g, min(max(R, 0.0), 1.0))
    h = _action_terms(B_self, max(B_opp, 0.0), S_self, S_opp, g, min(max(R, 0.0), 1.0))
    q = h @ (z @ _THETA[t_idx].reshape(N_Z, N_H))
    k = int(np.argmax(q))
    delta = float(np.sqrt(U_GRID[k] * B_self))
    W = B_self - delta**2
    x = float(np.sign(g) * V_GRID[k] * W)
    # guard against floating-point overspending (delta^2 + |x| <= B)
    while delta**2 > B_self:
        delta = float(np.nextafter(delta, 0.0))
    while delta**2 + abs(x) > B_self:
        x = float(np.nextafter(x, 0.0))
    return delta, x
