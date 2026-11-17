"""2.4 Refine your RL agents: off-policy control with value function approximation.

Methods (not used in 2.3):
  * Q-learning (Lecture 4, off-policy TD control), incremental semi-gradient
  * LSPI / LSTDQ (Lecture 5, batch off-policy control)
both with a wealth-scaled linear Q-function

    Q(s, a) = B_t * theta' phi(s, a),

where phi(s, a) = z(s) (x) h(s, a): the group's 50 state indicators z(s) from
features.py, interacted with a few economically motivated action terms h(s, a).
Because terminal wealth is (roughly) proportional to current wealth, the
Bellman equation divided by B_t reads

    theta' phi(s, a) = E[ rho * theta' phi(s', pi(s')) ],  rho = B_{t+1} / B_t,

with theta' phi(s, a) = E[rho] in the last period. This is still linear value
function approximation; the scaling only changes the features (B_t * phi).

Contents (in order):
  rollouts, summarise         play games with the group's environment.sim_game
  trade_gain, myopic_batch,   closed-form expected trading gain, one-step
  standard_pool, Mixture      look-ahead benchmark and the opponent types
  state_features_batch,       the group's features.state_features plus the six
  action_terms, greedy_batch  action terms; the greedy (epsilon-greedy) policy
  collect                     epsilon-greedy data for the replay buffer
  train_lspi                  LSPI with shared weights (oscillates; reported in
                              the write-up, not run by refine_experiments.py)
  fit_backward,               finite-horizon LSPI: the final agent
  train_backward
  train_qlearning             incremental Q-learning
  train_ac                    actor-critic network (needs PyTorch)

Policies take a dict of state arrays, a leftover of batch evaluation; with
sim_game they are always called with one state at a time.

Run refine_experiments.py (not this file) to train the agent / reproduce results,
and test_refine.py to check the code.
"""

import numpy as np
from scipy.special import ndtr

from default_opponent import default_opponent_policy
from environment import sim_game
from features import state_features


T = 10
SIGMA = 1.5
MAX_EFFECTIVE_SPEED = 350.0  # keep the supplied simulator's exp(2 * speed) finite
STATE_KEYS = ("t", "B_self", "B_opp", "S_self", "S_opp", "g", "R")


def rollouts(policy, opponent, n_games, seed=0, record=False):
    """Run the Section 2.1 simulator once per game and group the resulting data."""
    rng = np.random.default_rng(seed)
    budgets = np.empty((n_games, 2))
    if record:
        trajectories = np.empty((n_games, T + 1, 13))
        actions = np.empty((n_games, T), dtype=int)

    def for_environment(candidate, capture=False):
        if candidate is default_opponent_policy:
            return candidate

        def act(state):
            current = {key: np.array([value]) for key, value in zip(STATE_KEYS, state)}
            delta, x = candidate(current, rng)
            if capture:
                actions[game, state[0]] = candidate.last_idx[0]

            budget = state[1]
            delta = float(delta[0])
            while delta**2 > budget:
                delta = float(np.nextafter(delta, 0.0))
            remaining = budget - delta**2
            x = min(max(float(x[0]), -remaining), remaining)
            for _ in range(1000):          # a few rounding steps are normally enough
                if delta**2 + abs(x) <= budget:
                    break
                x = float(np.nextafter(x, 0.0))
            else:                          # |x| is negligibly small: trade nothing
                x = 0.0
            return [delta, x]

        return act

    own = for_environment(policy, capture=record)
    other = for_environment(opponent)
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        for game in range(n_games):
            history = sim_game(policy_1=own, policy_2=other)
            budgets[game] = history[-1, 1:3]
            if record:
                trajectories[game] = history
    finally:
        np.random.set_state(random_state)

    if not record:
        return budgets[:, 0], budgets[:, 1]

    log = []
    for t in range(T):
        rows = trajectories[:, t]
        log.append(dict(state=dict(zip(STATE_KEYS, rows[:, :7].T)),
                        idx=actions[:, t], delta=rows[:, 9], win=rows[:, 8] == 1,
                        B_next=trajectories[:, t + 1, 1]))
    return budgets[:, 0], budgets[:, 1], log


# Benchmark policies and payoff calculations


def hold_batch(st, rng):
    """Benchmark: never invest in speed, never trade (keeps B = 100)."""
    z = np.zeros_like(st["B_self"])
    return z, z


def summarise(B1, B2):
    """Summary of firm 1's terminal budgets: mean, standard error, median, win rate."""
    n = len(B1)
    return dict(mean=float(B1.mean()), se=float(B1.std(ddof=1) / np.sqrt(n)),
                median=float(np.median(B1)), win_rate=float(np.mean(B1 > B2)),
                opp_mean=float(B2.mean()))


DELTA_GRID = np.linspace(0, 12, 241)
Z_NODES = (np.arange(20) + 0.5) / 20  # quadrature for opponent's Z ~ U[0,1]


def trade_gain(g, R, f=1.0):
    """E[max(0, 1 + f*sign(g)*Delta)] - 1 per unit of wealth, given g, R.

    Given the signal, f*sign(g)*Delta ~ N(f|g|, f^2 (1-R^2) sigma^2), so
    Y = 1 + f*sign(g)*Delta is normal with mean mu and s.d. s, and
    E[max(0, Y)] = mu * Phi(mu/s) + s * phi(mu/s)  (mean of a normal floored at 0).
    """
    g = np.asarray(g, float)
    R = np.asarray(R, float)
    mu = 1 + f * np.abs(g)
    s = np.maximum(f * SIGMA * np.sqrt(np.maximum(1 - R**2, 0)), 1e-9)
    z = mu / s
    return mu * ndtr(z) + s * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) - 1


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
        # one-period expected profit: P(win) * W * gain - speed cost (+ optional speed value)
        value = p * W * G - cost + speed_value * DELTA_GRID[:, None]
        value = np.where(cost <= B[None, :], value, -np.inf)   # unaffordable speeds excluded
        k = np.argmax(value, axis=0)           # best delta on the grid, per game
        delta = DELTA_GRID[k]
        W = np.maximum(B - delta**2, 0)
        x = np.sign(g) * trade_f * W
        return delta, x
    return policy


def allin_nospeed(st, rng=None):
    """Opponent type: no speed investment, bets the whole budget on sign(g)."""
    z = np.zeros_like(st["B_self"])
    return z, np.sign(st["g"]) * st["B_self"]


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
        """pool: list of batched policies; weights: relative sampling weights."""
        self.pool = pool
        self.w = np.ones(len(pool)) / len(pool) if weights is None else np.asarray(weights) / np.sum(weights)
        self.assign = None

    def __call__(self, st, rng):
        """Act for every game with that game's assigned opponent."""
        n = len(st["B_self"])
        # draw a fresh assignment at the start of each batch of games
        if int(st["t"][0]) == 0 or self.assign is None or len(self.assign) != n:
            self.assign = rng.choice(len(self.pool), size=n, p=self.w)
        d = np.zeros(n); x = np.zeros(n)
        for k, pol in enumerate(self.pool):
            m = self.assign == k
            if m.any():
                sub = {kk: v[m] for kk, v in st.items()}
                if pol is default_opponent_policy:
                    states = zip(*(sub[key] for key in STATE_KEYS))
                    d[m], x[m] = np.array([pol(list(state)) for state in states]).T
                else:
                    d[m], x[m] = pol(sub, rng)
        return d, x


def standard_pool():
    """Opponent types used for the robustness checks and pool training."""
    return {
        "default": default_opponent_policy,
        "hold cash": hold_batch,
        "all-in, no speed": allin_nospeed,
        "one-step look-ahead": myopic_batch("default"),
        "const speed 2": const_speed(2.0),
        "const speed 5": const_speed(5.0),
    }


# ---------------------------------------------------------------------------
# Action grid (finer than 2.3 at small speed spends: delta = 1 costs only 1%)
# ---------------------------------------------------------------------------
SPEED_FRACS = np.array([0, .0025, .005, .01, .02, .03, .05, .075, .1, .15,
                        .2, .3, .4, .5, .7, 1.0])
TRADE_FRACS = np.linspace(0, 1, 11)
U_GRID, V_GRID = [a.ravel() for a in np.meshgrid(SPEED_FRACS, TRADE_FRACS,
                                                  indexing="ij")]
N_ACTIONS = len(U_GRID)


# ---------------------------------------------------------------------------
# State features: same 50 indicators as features.state_features, vectorised
# ---------------------------------------------------------------------------
def state_features_batch(st):
    """The group's 50 state indicators (features.state_features) for a batch of states."""
    rows = zip(*(st[k] for k in ("t", "B_self", "B_opp", "S_self", "S_opp", "g", "R")))
    return np.array([state_features(list(r)) for r in rows], dtype=float).reshape(-1, 50)


# ---------------------------------------------------------------------------
# Action terms h(s, a)
# ---------------------------------------------------------------------------
H_NAMES = ["kept wealth (1-u)",
           "(1-u) * P_win(default opp) * gain(v)",
           "(1-u) * P_win(passive opp) * gain(v)",
           "(1-u) * P_win(default opp)  [race position]",
           "(1-u) * P_win(passive opp)  [race position]",
           "(1-u) * v * P_win(default opp)  [bet size]"]


# "economic": action terms built from the game rules (final agent).
# "plain": the group's 2.3 quadratic action terms 1, u, v, u^2, v^2, uv (ablation).
FEATURE_SET = "economic"


def _speed_investment(budget, speed, fraction):
    return np.minimum(np.sqrt(fraction * budget), np.maximum(MAX_EFFECTIVE_SPEED - speed, 0))


def action_terms(st, u, v):
    """h(s, a) for every state (n) and every candidate action (k): shape (n, k, H)."""
    B = st["B_self"][:, None]
    delta = _speed_investment(B, st["S_self"][:, None], u[None, :])
    spend = np.divide(delta**2, B, out=np.zeros_like(delta), where=B > 0)
    if FEATURE_SET == "plain":
        n = len(st["B_self"])
        uu = spend
        vv = np.broadcast_to(v[None, :], (n, len(v)))
        return np.stack([np.ones_like(uu), uu, vv, uu * uu, vv * vv, uu * vv], axis=2)
    kept = 1 - spend
    # expected one-period return per unit of wealth when betting share f on sign(g)
    gain = trade_gain(st["g"][:, None], st["R"][:, None], TRADE_FRACS)
    gv = gain[:, np.round(v * 10).astype(int)]  # pick the gain for each action's trade share v
    # win probabilities against a default-like and a passive opponent
    p_def = _p_win(delta, st, np.sqrt(st["B_opp"]))
    p_pas = _p_win(delta, st, np.zeros_like(st["B_opp"]))
    return np.stack([kept, kept * p_def * gv, kept * p_pas * gv,
                     kept * p_def, kept * p_pas, kept * v[None, :] * p_def], axis=2)


_Z = (np.arange(20) + 0.5) / 20


def _p_win(delta, st, opp_scale):
    """P(win) when opponent invests Z*opp_scale, Z ~ U[0,1]; delta shape (n, k).

    The softmax lottery gives P(win) = sigmoid(2 * (own speed - opponent speed));
    the expectation over the opponent's random Z uses 20 quadrature points (_Z).
    opp_scale = sqrt(B_opp) mimics the default opponent, 0 a passive opponent.
    """
    z = 2 * (st["S_self"][:, None, None] + delta[:, :, None]
             - st["S_opp"][:, None, None] - opp_scale[:, None, None] * _Z)
    return (1 / (1 + np.exp(-np.clip(z, -50, 50)))).mean(axis=2)


N_H = len(H_NAMES)
N_Z = 50
N_PARAMS = N_H * N_Z


def phi_all(st, u=U_GRID, v=V_GRID):
    """Features for every candidate action: returns z (n,50) and h (n,k,H).

    phi(s,a) = vec(z(s) outer h(s,a)); Q/B = z' Theta h with Theta (50 x H).
    """
    return state_features_batch(st), action_terms(st, u, v)


def q_all(theta, z, h):
    """Normalised Q (per unit of wealth) for all actions: (n, k)."""
    coef = z @ theta.reshape(N_Z, N_H)          # (n, H)
    return np.einsum("nh,nkh->nk", coef, h)


def phi_of(z, h_sel):
    """Flattened feature vector(s) for chosen actions: (n, 50*H)."""
    return (z[:, :, None] * h_sel[:, None, :]).reshape(len(z), -1)


def actions_from_index(st, idx):
    """Convert grid indices to the game's actions (delta, x).

    delta = sqrt(u * B) so that the speed cost delta^2 is a share u of the budget;
    x bets a share v of the post-cost wealth W = B - delta^2 in the direction of g.
    """
    u, v = U_GRID[idx], V_GRID[idx]
    B = st["B_self"]
    delta = _speed_investment(B, st["S_self"], u)
    x = np.sign(st["g"]) * v * (B - delta**2)
    return delta, x


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def greedy_batch(theta, epsilon=0.0):
    """Batched (epsilon-)greedy policy w.r.t. the linear Q-function with weights theta."""
    def policy(st, rng):
        n, chunk = len(st["B_self"]), 4000
        if n == 1:
            if st["B_self"][0] == 0:
                policy.last_idx = np.zeros(1, dtype=int)
                return actions_from_index(st, policy.last_idx)
            if epsilon and rng.random() < epsilon:
                policy.last_idx = rng.integers(N_ACTIONS, size=1)
                return actions_from_index(st, policy.last_idx)

        idx = np.empty(n, dtype=int)
        for s in range(0, n, chunk):           # chunks keep memory small for many games
            z, h = phi_all(_sub(st, np.arange(n)[s:s + chunk]))
            idx[s:s + chunk] = np.argmax(q_all(_th(theta, st), z, h), axis=1)
        if epsilon > 0 and n > 1:              # single-state exploration is handled above
            explore = rng.uniform(size=len(idx)) < epsilon
            idx = np.where(explore, rng.integers(0, N_ACTIONS, len(idx)), idx)
        policy.last_idx = idx                  # remembered so collect() can log the actions
        return actions_from_index(st, idx)
    return policy


# ---------------------------------------------------------------------------
# Data collection (experience replay buffer of transitions)
# ---------------------------------------------------------------------------
def collect(theta, epsilon, opponent, n_games, seed):
    """Play n_games with an epsilon-greedy behaviour policy; return transitions.

    Each transition: (z, h_chosen, rho, z', h'(all actions), terminal).
    States with zero wealth are absorbing (value 0) and dropped.
    """
    # first round (theta None): a purely random behaviour policy
    pol = greedy_batch(theta if theta is not None else np.zeros(N_PARAMS),
                       epsilon if theta is not None else 1.0)
    B1, B2, log = rollouts(pol, opponent, n_games, seed=seed, record=True)
    data = []
    for t in range(T):
        st = log[t]["state"]
        alive = st["B_self"] > 0
        sa = _sub(st, alive)
        idx = log[t]["idx"][alive]
        h_sel = _h_chosen(sa, idx)
        rho = log[t]["B_next"][alive] / st["B_self"][alive]   # wealth ratio B_{t+1} / B_t
        item = dict(z=state_features_batch(sa).astype(np.float32),
                    h=h_sel.astype(np.float32), rho=rho, terminal=t == T - 1, t=t)
        if t < T - 1:
            # store the raw next state; features of all next actions are recomputed
            # in chunks when needed (keeps memory small)
            st_next = _sub(log[t + 1]["state"], alive)
            item["ok"] = st_next["B_self"] > 0
            item["next"] = _sub(st_next, item["ok"])
        data.append(item)
    return data, summarise(B1, B2)


def _h_chosen(st, idx, chunk=4000):
    """h(s, a) for the chosen action of each state."""
    out = np.empty((len(idx), N_H))
    for s in range(0, len(idx), chunk):
        sl = slice(s, s + chunk)
        out[sl] = action_terms(_sub(st, np.arange(len(idx))[sl]), U_GRID, V_GRID)[
            np.arange(len(idx[sl])), idx[sl]]
    return out


def greedy_next(theta, st, chunk=4000):
    """For next states: features of the greedy action and its normalised Q."""
    n = len(st["B_self"])
    phi = np.zeros((n, N_PARAMS), dtype=np.float32)
    qmax = np.zeros(n)
    for s in range(0, n, chunk):
        sub = _sub(st, np.arange(n)[s:s + chunk])
        z, h = phi_all(sub)
        q = q_all(_th(theta, sub), z, h)
        best = np.argmax(q, axis=1)
        qmax[s:s + chunk] = q[np.arange(len(best)), best]
        phi[s:s + chunk] = phi_of(z, h[np.arange(len(best)), best])
    return phi, qmax


def _th(theta, st):
    """Shared weights (1-D) or period-specific weights (T x params, 2-D)."""
    if theta.ndim == 1:
        return theta
    return theta[int(st["t"][0])] if len(st["t"]) else theta[0]


def _sub(st, mask):
    """Select the games in `mask` from a dict of state arrays."""
    return {k: v[mask] for k, v in st.items()}


# ---------------------------------------------------------------------------
# LSPI with LSTDQ (Lecture 5, batch control)
# ---------------------------------------------------------------------------
def lstdq(data, theta, ridge=1e-3):
    """Solve sum phi (phi - rho phi'_greedy)' theta = sum phi rho*[terminal]."""
    A = ridge * np.eye(N_PARAMS)
    b = np.zeros(N_PARAMS)
    for d in data:
        phi = phi_of(d["z"], d["h"]).astype(float)
        target_phi = np.zeros_like(phi)
        if d["terminal"]:
            b += phi.T @ d["rho"]
        elif d["ok"].any():
            target_phi[d["ok"]], _ = greedy_next(theta, d["next"])
        A += phi.T @ (phi - d["rho"][:, None] * target_phi)
    return np.linalg.solve(A, b)


def train_lspi(opponent=default_opponent_policy, n_rounds=8, games_per_round=4000,
               epsilon=0.2, buffer_rounds=4, seed=0, eval_games=20000, verbose=True):
    """LSPI: alternate data collection (epsilon-greedy) and LSTDQ policy iteration."""
    theta = None
    buffer, curve = [], []
    for r in range(n_rounds):
        data, _ = collect(theta, epsilon, opponent, games_per_round, seed + 1000 * r)
        buffer = (buffer + [data])[-buffer_rounds:]
        flat = [d for chunk in buffer for d in chunk]
        th = np.zeros(N_PARAMS) if theta is None else theta
        for _ in range(3):                     # policy-iteration sweeps on the buffer
            th = lstdq(flat, th)
        theta = th
        res = evaluate(theta, opponent, eval_games, seed=777)
        curve.append(res)
        if verbose:
            print(f"LSPI round {r + 1}/{n_rounds}: greedy mean B10 = {res['mean']:.1f} "
                  f"(median {res['median']:.1f}, win {res['win_rate']:.0%})", flush=True)
    return theta, curve


# ---------------------------------------------------------------------------
# Finite-horizon LSPI: period-specific weights => LSTDQ solves backwards
# ---------------------------------------------------------------------------
def fit_backward(buffer, ridge=1.0, theta_init=None):
    """LSTDQ with period-specific weights theta_t.

    With one weight block per period, the LSTDQ matrix is block-triangular
    (period t only bootstraps from period t+1), so the fixed point is found
    exactly by backward induction: OLS for t = 9 on rho, then for t = 8 on
    rho * max_a' Q_9(s', a'), ... This is batch Q-learning / approximate DP.
    """
    theta = np.zeros((T, N_PARAMS)) if theta_init is None else theta_init.copy()
    for t in reversed(range(T)):               # t = 9, 8, ..., 0
        A = ridge * np.eye(N_PARAMS)           # ridge-regularised normal equations
        b = np.zeros(N_PARAMS)
        for chunk in buffer:                   # all rounds of data in the replay buffer
            d = chunk[t]
            phi = phi_of(d["z"], d["h"]).astype(float)
            if d["terminal"]:
                y = d["rho"]                   # last period: target is B_10 / B_9
            else:
                # Q-learning target: rho * max_a' Q_{t+1}(s', a'), using the weights
                # theta[t+1] fitted in the previous step; bankrupt next states are worth 0
                nxt = np.zeros(len(phi))
                if d["ok"].any():
                    _, nxt[d["ok"]] = greedy_next(theta, d["next"])
                y = d["rho"] * nxt
            A += phi.T @ phi
            b += phi.T @ y
        theta[t] = np.linalg.solve(A, b)       # OLS (ridge) fit for period t
    return theta


def train_backward(opponent=default_opponent_policy, n_rounds=6, games_per_round=8000,
                   epsilon=0.2, buffer_rounds=6, ridge=1.0, seed=0,
                   eval_games=20000, verbose=True, opponent_factory=None):
    """Finite-horizon LSPI: collect epsilon-greedy data, refit backwards, repeat.

    opponent_factory(theta) -> batched opponent policy, if given, is called every
    round (used for training against a pool that includes the current agent).
    Evaluation is always against `opponent`.
    """
    theta, buffer, curve = None, [], []
    for r in range(n_rounds):
        train_opp = opponent if opponent_factory is None else opponent_factory(theta)
        # 1) new data from the current epsilon-greedy policy
        data, _ = collect(theta, epsilon, train_opp, games_per_round, seed + 1000 * r)
        # 2) keep the last `buffer_rounds` rounds (off-policy, so old data stays valid)
        buffer = (buffer + [data])[-buffer_rounds:]
        # 3) policy improvement: refit all periods backwards on the whole buffer
        theta = fit_backward(buffer, ridge)
        res = evaluate(theta, opponent, eval_games, seed=777)
        curve.append(res)
        if verbose:
            print(f"Backward-LSPI round {r + 1}/{n_rounds}: greedy mean B10 = "
                  f"{res['mean']:.1f} ± {res['se']:.1f} (median {res['median']:.1f}, "
                  f"win {res['win_rate']:.0%})", flush=True)
    return theta, curve


# ---------------------------------------------------------------------------
# Incremental Q-learning (Lecture 4 + semi-gradient linear VFA, Lecture 5)
# ---------------------------------------------------------------------------
def train_qlearning(opponent=default_opponent_policy, n_batches=60, games_per_batch=500,
                    alpha0=0.02, eps0=0.3, eps_min=0.05, seed=0, eval_every=10,
                    eval_games=20000, verbose=True):
    """Semi-gradient Q-learning on the normalised Q, updates per transition batch.

    alpha_k = alpha0 / (1 + k/20)^0.6, epsilon_k = max(eps_min, eps0/(1+k/10)).
    """
    theta = np.zeros(N_PARAMS)
    curve = []
    for k in range(n_batches):
        alpha = alpha0 / (1 + k / 20) ** 0.6
        eps = max(eps_min, eps0 / (1 + k / 10))
        data, _ = collect(theta if k else None, eps, opponent, games_per_batch, seed + k)
        for d in reversed(data):                       # backwards = faster propagation
            phi = phi_of(d["z"], d["h"])
            pred = phi @ theta                         # current normalised Q(s, a)
            if d["terminal"]:
                target = d["rho"]
            else:
                nxt = np.zeros(len(phi))
                if d["ok"].any():
                    _, nxt[d["ok"]] = greedy_next(theta, d["next"])
                target = d["rho"] * nxt
            err = np.clip(target - pred, -5, 5)        # clip rare huge ratios
            theta += alpha * phi.T @ err / len(phi)    # semi-gradient step, averaged over batch
        if (k + 1) % eval_every == 0:
            res = evaluate(theta, opponent, eval_games, seed=777)
            curve.append((k + 1, res))
            if verbose:
                print(f"Q-learning batch {k + 1}/{n_batches}: greedy mean B10 = "
                      f"{res['mean']:.1f} (median {res['median']:.1f})", flush=True)
    return theta, curve


def evaluate(theta, opponent, n_games=20000, seed=777):
    """Mean/median terminal budget of the greedy policy over n_games (fixed seed)."""
    B1, B2 = rollouts(greedy_batch(theta), opponent, n_games, seed=seed)
    return summarise(B1, B2)


# ---------------------------------------------------------------------------
# Actor-critic with a neural network (Lecture 6, mini-problem set 6).
# Needs PyTorch; only used in the experiments, not by the submitted agent.
# ---------------------------------------------------------------------------
AC_N_IN = 59


def ac_inputs(st):
    """Continuous, scaled state variables plus the group's 50 indicators."""
    cont = np.stack([st["t"] / 9, np.log1p(st["B_self"]) / 5, np.log1p(st["B_opp"]) / 5,
                     st["S_self"] / 10, st["S_opp"] / 10, (st["S_self"] - st["S_opp"]) / 5,
                     st["g"] / 1.5, np.abs(st["g"]) / 1.5, st["R"]], axis=1)
    return np.hstack([cont, state_features_batch(st)]).astype(np.float32)


def _actor_critic_class():
    """Build the network class lazily so numpy-only users never import PyTorch."""
    import torch.nn as nn
    import torch.nn.functional as F

    class ActorCritic(nn.Module):
        """Shared layer, actor head (action preferences) and critic head, as in PS6."""

        def __init__(self, n_in=AC_N_IN, hidden=128, n_actions=N_ACTIONS):
            """Shared hidden layer, then one logit per action and one state value."""
            super().__init__()
            self.common = nn.Linear(n_in, hidden)
            self.actor = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            """Return (action logits, state value) for a batch of inputs."""
            h = F.relu(self.common(x))
            return self.actor(h), self.critic(h).squeeze(-1)

    return ActorCritic


def ac_policy(model, greedy=False):
    """Batched policy from the actor: argmax (greedy) or sampled from the softmax."""
    import torch

    def policy(st, rng):
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(ac_inputs(st)))
            if greedy:
                idx = logits.argmax(1).numpy()
            else:
                p = torch.softmax(logits, 1).numpy().astype(float)
                p /= p.sum(1, keepdims=True)
                c = p.cumsum(1)                        # inverse-CDF sampling per game
                idx = (rng.uniform(size=(len(p), 1)) < c).argmax(1)
        policy.last_idx = idx
        return actions_from_index(st, idx)
    return policy


def train_ac(n_updates=12000, games=512, lr=1e-3, entropy=0.003, clip_return=1000.0,
             seed=0, opponent=default_opponent_policy, eval_every=1200, eval_games=10000, verbose=True):
    """Advantage actor-critic with Monte-Carlo returns per unit of wealth, rho = B_T / B_t.

    Mirrors mini-problem set 6: shared layer + actor head (softmax action
    preferences over the 176-action grid) + critic head; standardised advantages,
    Huber critic loss, small entropy bonus; Adam with decreasing step size.
    Unlike LSPI it sees only the raw state (no rules-based action features).
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = _actor_critic_class()()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # decreasing step size: lr_k = lr / (1 + k / (n_updates/5))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1 / (1 + 5 * k / n_updates))
    curve = []
    for k in range(n_updates):
        # 1) play a batch of games with the current stochastic policy
        pol = ac_policy(model)
        _, _, log = rollouts(pol, opponent, games, seed=seed * 100000 + k, record=True)
        # 2) Monte-Carlo return of every visited state, per unit of wealth: B_T / B_t
        BT = log[-1]["B_next"]
        X, A, G = [], [], []
        for t in range(T):
            st = log[t]["state"]
            alive = st["B_self"] > 0
            X.append(ac_inputs({kk: v[alive] for kk, v in st.items()}))
            A.append(log[t]["idx"][alive])
            G.append(np.minimum(BT[alive] / st["B_self"][alive], clip_return))  # return per unit of wealth
        X = torch.from_numpy(np.vstack(X)); A = torch.from_numpy(np.concatenate(A)).long()
        G = torch.from_numpy(np.concatenate(G).astype(np.float32))
        # 3) losses: advantage-weighted log-probabilities (actor), Huber (critic)
        logits, V = model(X)
        logp = torch.log_softmax(logits, 1)
        adv = (G - V).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)                  # standardise, as in PS6
        actor_loss = -(logp.gather(1, A[:, None]).squeeze(1) * adv).mean()
        critic_loss = F.huber_loss(V, G)
        ent = -(logp.exp() * logp).sum(1).mean()       # entropy bonus keeps exploring
        loss = actor_loss + critic_loss - entropy * ent
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (k + 1) % eval_every == 0:
            r = summarise(*rollouts(ac_policy(model, greedy=True), opponent, eval_games, seed=777))
            curve.append((k + 1, r))
            if verbose:
                print(f"AC update {k + 1}: greedy mean B10 {r['mean']:.0f} ± {r['se']:.0f}, "
                      f"entropy {ent.item():.2f}", flush=True)
    return model, curve
