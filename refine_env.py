"""2.4 support code: fast simulator, benchmark policies and opponent pool.

Part 1 is a vectorised copy of environment.sim_game for fast training/evaluation.

Plays many games in parallel with NumPy. Dynamics are identical to
environment.sim_game (checked in check_fastsim()): sigma = 1.5, softmax speed
lottery with factor 2, decay d = U^2 with U ~ U[0, sqrt(0.1)], budget floor at 0.

Batched policies take a dict of arrays
    {"t", "B_self", "B_opp", "S_self", "S_opp", "g", "R"}  (each shape (n,))
and return (delta, x) arrays of shape (n,).
"""

import numpy as np
from scipy.stats import norm

SIGMA = 1.5
T = 10


def default_batch(st, rng):
    """Vectorised default opponent (identical rule to default_opponent.py)."""
    B = st["B_self"]
    delta = rng.uniform(0, 1, B.shape) * np.sqrt(B)
    W = B - delta**2
    x = W * np.clip(st["g"] / 2, -1, 1)
    return delta, x


def hold_batch(st, rng):
    z = np.zeros_like(st["B_self"])
    return z, z


def _feasible(delta, x, B):
    delta = np.clip(np.nan_to_num(delta), 0, np.sqrt(B))
    W = np.maximum(B - delta**2, 0)
    x = np.clip(np.nan_to_num(x), -W, W)
    return delta, x, W


def play(policy_1, policy_2, n_games, seed=0, record=False, callback=None):
    """Simulate n_games games in parallel. Returns terminal budgets (B1, B2).

    record=True also returns a list of per-period dicts (states, actions, winner).
    callback(t, state_1, (delta1, x1), win1, B1_next) is called each period
    (used by learning algorithms to collect transitions).
    """
    rng = np.random.default_rng(seed)
    B1 = np.full(n_games, 100.0)
    B2 = np.full(n_games, 100.0)
    S1 = np.zeros(n_games)
    S2 = np.zeros(n_games)
    log = []
    for t in range(T):
        R = rng.uniform(0, 1, n_games)
        g = rng.normal(0, 1, n_games) * R * SIGMA
        tt = np.full(n_games, t)
        st1 = dict(t=tt, B_self=B1, B_opp=B2, S_self=S1, S_opp=S2, g=g, R=R)
        st2 = dict(t=tt, B_self=B2, B_opp=B1, S_self=S2, S_opp=S1, g=g, R=R)
        d1, x1 = policy_1(st1, rng)
        d2, x2 = policy_2(st2, rng)
        d1, x1, W1 = _feasible(d1, x1, B1)
        d2, x2, W2 = _feasible(d2, x2, B2)
        e1, e2 = S1 + d1, S2 + d2
        p1 = 1.0 / (1.0 + np.exp(np.clip(2 * (e2 - e1), -700, 700)))
        win1 = rng.uniform(0, 1, n_games) < p1
        eps = rng.normal(0, 1, n_games) * SIGMA * np.sqrt(1 - R**2)
        move = g + eps
        nB1 = np.where(win1, np.maximum(0, W1 + x1 * move), W1)
        nB2 = np.where(~win1, np.maximum(0, W2 + x2 * move), W2)
        if record:
            log.append(dict(state=st1, delta=d1, x=x1, win=win1, p_win=p1,
                            delta_opp=d2, B_next=nB1))
        if callback is not None:
            callback(t, st1, (d1, x1), win1, nB1)
        U1 = rng.uniform(0, np.sqrt(0.1), n_games)
        U2 = rng.uniform(0, np.sqrt(0.1), n_games)
        S1 = e1 * (1 - U1**2)
        S2 = e2 * (1 - U2**2)
        B1, B2 = nB1, nB2
    if record:
        return B1, B2, log
    return B1, B2


def play_chunked(policy_1, policy_2, n_games, seed=0, chunk=5000):
    """Same as play() but in chunks to bound memory; returns (B1, B2)."""
    out1, out2 = [], []
    for i, start in enumerate(range(0, n_games, chunk)):
        b1, b2 = play(policy_1, policy_2, min(chunk, n_games - start), seed=seed * 1000 + i)
        out1.append(b1)
        out2.append(b2)
    return np.concatenate(out1), np.concatenate(out2)


def summarise(B1, B2):
    n = len(B1)
    return dict(mean=float(B1.mean()), se=float(B1.std(ddof=1) / np.sqrt(n)),
                median=float(np.median(B1)), win_rate=float(np.mean(B1 > B2)),
                opp_mean=float(B2.mean()))


def check_fastsim(n=20000):
    """Compare default-vs-default and hold-vs-default with environment.sim_game."""
    from environment import sim_game
    from default_opponent import default_opponent_policy as D
    out = {}
    for name, fast_p, slow_p in [("default", default_batch, D),
                                 ("hold", hold_batch, lambda s: [0.0, 0.0])]:
        fb1, fb2 = play(fast_p, default_batch, n, seed=1)
        np.random.seed(1)
        res = np.array([sim_game(policy_1=slow_p, policy_2=D)[-1, 1:3]
                        for _ in range(n // 4)])
        out[name] = dict(fast=(fb1.mean(), fb2.mean()),
                         slow=(res[:, 0].mean(), res[:, 1].mean()))
    return out


# ===========================================================================
# Part 2: model-based benchmark (one-step look-ahead) and closed-form trade gain
# ===========================================================================
DELTA_GRID = np.linspace(0, 12, 241)
Z_NODES = (np.arange(20) + 0.5) / 20  # quadrature for opponent's Z ~ U[0,1]


def trade_gain(g, R, f=1.0):
    """E[max(0, 1 + f*sign(g)*Delta)] - 1 per unit of wealth, given g, R."""
    g = np.asarray(g, float)
    R = np.asarray(R, float)
    mu = 1 + f * np.abs(g)
    s = np.maximum(f * SIGMA * np.sqrt(np.maximum(1 - R**2, 0)), 1e-9)
    return mu * norm.cdf(mu / s) + s * norm.pdf(mu / s) - 1


def win_prob(delta, S_self, S_opp, delta_opp_scale):
    """P(win) for own delta when the opponent invests Z*delta_opp_scale, Z~U[0,1]."""
    d = np.asarray(delta)[:, None, None]
    opp = S_opp[None, :, None] + delta_opp_scale[None, :, None] * Z_NODES[None, None, :]
    z = 2 * (S_self[None, :, None] + d - opp)
    return (1 / (1 + np.exp(-np.clip(z, -50, 50)))).mean(axis=2)


def myopic_batch(opp_model="default", trade_f=1.0, speed_value=0.0):
    """One-step optimal speed investment with all-in trading in direction of g.

    opp_model: 'default' assumes opponent delta = Z*sqrt(B_opp);
               'zero' assumes the opponent invests nothing this period.
    speed_value: shadow value per unit of speed capital carried forward.
    """
    def policy(st, rng=None):
        B, g, R = st["B_self"], st["g"], st["R"]
        if opp_model == "default":
            scale = np.sqrt(st["B_opp"])
        else:
            scale = np.zeros_like(B)
        p = win_prob(DELTA_GRID, st["S_self"], st["S_opp"], scale)       # (k, n)
        cost = DELTA_GRID[:, None] ** 2
        W = np.maximum(B[None, :] - cost, 0)
        G = trade_gain(g, R, trade_f)[None, :]
        value = p * W * G - cost + speed_value * DELTA_GRID[:, None]
        value = np.where(cost <= B[None, :], value, -np.inf)
        k = np.argmax(value, axis=0)
        delta = DELTA_GRID[k]
        W = np.maximum(B - delta**2, 0)
        x = np.sign(g) * trade_f * W
        return delta, x
    return policy


def allin_nospeed(st, rng=None):
    z = np.zeros_like(st["B_self"])
    return z, np.sign(st["g"]) * st["B_self"]


# ===========================================================================
# Part 3: opponent pool (robustness checks and pool training)
# ===========================================================================
def const_speed(delta_level, trade_f=1.0):
    """Invest a fixed delta every period (if affordable) and bet trade_f of wealth on sign(g)."""
    def policy(st, rng):
        B = st["B_self"]
        d = np.minimum(delta_level, np.sqrt(B))
        return d, np.sign(st["g"]) * trade_f * (B - d**2)
    return policy


class Mixture:
    """Each game is assigned one opponent from the pool for its whole duration."""
    def __init__(self, pool, weights=None):
        self.pool = pool
        self.w = np.ones(len(pool)) / len(pool) if weights is None else np.asarray(weights) / np.sum(weights)
        self.assign = None

    def __call__(self, st, rng):
        n = len(st["B_self"])
        if int(st["t"][0]) == 0 or self.assign is None or len(self.assign) != n:
            self.assign = rng.choice(len(self.pool), size=n, p=self.w)
        d = np.zeros(n); x = np.zeros(n)
        for k, pol in enumerate(self.pool):
            m = self.assign == k
            if m.any():
                sub = {kk: v[m] for kk, v in st.items()}
                d[m], x[m] = pol(sub, rng)
        return d, x


def standard_pool():
    return {
        "default": default_batch,
        "hold cash": hold_batch,
        "all-in, no speed": allin_nospeed,
        "one-step look-ahead": myopic_batch("default"),
        "const speed 2": const_speed(2.0),
        "const speed 5": const_speed(5.0),
    }


if __name__ == "__main__":
    for k, v in check_fastsim().items():
        print(k, v)
