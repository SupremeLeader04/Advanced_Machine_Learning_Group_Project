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

Also contains train_lspi (shared weights across periods), which oscillated
between rounds and is kept for the write-up's discussion, and train_backward,
the finite-horizon LSPI used for the final agent.

Run refine_experiments.py (not this file) to train the agent / reproduce results.
"""

import numpy as np

from refine_env import T, default_batch, play, play_chunked, summarise, trade_gain


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
def _onehot(idx, k):
    out = np.zeros((len(idx), k))
    out[np.arange(len(idx)), idx] = 1
    return out


def state_features_batch(st):
    """Vectorised version of features.state_features (identical bins)."""
    t = st["t"].astype(int)
    d = st["S_self"] - st["S_opp"]
    sp = np.select([d < -5, d < -3, d < 0, d == 0, d < 3, d < 5],
                   [0, 1, 2, 3, 4, 5], 6)
    g = st["g"]
    return np.hstack([
        _onehot(t // 2, 5),
        _onehot(np.searchsorted([20, 40, 60, 80, 100, 120, 150], st["B_self"], "right"), 8),
        _onehot(np.searchsorted([20, 40, 60, 80, 100, 120, 150], st["B_opp"], "right"), 8),
        _onehot(sp, 7),
        _onehot(np.searchsorted(np.arange(1, 10) / 10, st["R"], "right"), 10),
        _onehot(np.select([g < 0, g == 0], [0, 1], 2), 3),
        _onehot(np.searchsorted([.5, 1, 1.5, 2, 2.5, 3, 4, 10], np.abs(g), "right"), 9),
    ])


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


def action_terms(st, u, v):
    """h(s, a) for every state (n) and every candidate action (k): shape (n, k, H)."""
    if FEATURE_SET == "plain":
        n = len(st["B_self"])
        uu = np.broadcast_to(u[None, :], (n, len(u)))
        vv = np.broadcast_to(v[None, :], (n, len(v)))
        return np.stack([np.ones_like(uu), uu, vv, uu * uu, vv * vv, uu * vv], axis=2)
    B = st["B_self"][:, None]
    delta = np.sqrt(u[None, :] * B)                                   # (n, k)
    kept = np.broadcast_to(1 - u[None, :], delta.shape)
    gain = np.stack([trade_gain(st["g"], st["R"], f) for f in TRADE_FRACS], 1)  # (n, 11)
    gv = gain[:, np.round(v * 10).astype(int)]                         # (n, k)
    # win probabilities against a default-like and a passive opponent
    p_def = _p_win(delta, st, np.sqrt(st["B_opp"]))
    p_pas = _p_win(delta, st, np.zeros_like(st["B_opp"]))
    return np.stack([kept, kept * p_def * gv, kept * p_pas * gv,
                     kept * p_def, kept * p_pas, kept * v[None, :] * p_def], axis=2)


_Z = (np.arange(20) + 0.5) / 20


def _p_win(delta, st, opp_scale):
    """P(win) when opponent invests Z*opp_scale, Z ~ U[0,1]; delta shape (n, k)."""
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
    u, v = U_GRID[idx], V_GRID[idx]
    B = st["B_self"]
    delta = np.sqrt(u * B)
    x = np.sign(st["g"]) * v * (B - delta**2)
    return delta, x


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def greedy_batch(theta, epsilon=0.0):
    def policy(st, rng):
        n, chunk = len(st["B_self"]), 4000
        idx = np.empty(n, dtype=int)
        for s in range(0, n, chunk):
            z, h = phi_all(_sub(st, np.arange(n)[s:s + chunk]))
            idx[s:s + chunk] = np.argmax(q_all(_th(theta, st), z, h), axis=1)
        if epsilon > 0:
            explore = rng.uniform(size=len(idx)) < epsilon
            idx = np.where(explore, rng.integers(0, N_ACTIONS, len(idx)), idx)
        policy.last_idx = idx
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
    pol = greedy_batch(theta if theta is not None else np.zeros(N_PARAMS),
                       epsilon if theta is not None else 1.0)
    idx_log = []

    def cb(t, st, a, win, B_next):
        idx_log.append(pol.last_idx.copy())

    B1, B2, log = play(pol, opponent, n_games, seed=seed, record=True, callback=cb)
    data = []
    for t in range(T):
        st = log[t]["state"]
        alive = st["B_self"] > 0
        sa = _sub(st, alive)
        idx = idx_log[t][alive]
        h_sel = _h_chosen(sa, idx)
        rho = log[t]["B_next"][alive] / st["B_self"][alive]
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


def train_lspi(opponent=default_batch, n_rounds=8, games_per_round=4000,
               epsilon=0.2, buffer_rounds=4, seed=0, eval_games=20000, verbose=True):
    """LSPI: alternate data collection (epsilon-greedy) and LSTDQ policy iteration."""
    theta = None
    buffer, curve = [], []
    for r in range(n_rounds):
        data, behav = collect(theta, epsilon, opponent, games_per_round, seed + 1000 * r)
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
    for t in reversed(range(T)):
        A = ridge * np.eye(N_PARAMS)
        b = np.zeros(N_PARAMS)
        for chunk in buffer:
            d = chunk[t]
            phi = phi_of(d["z"], d["h"]).astype(float)
            if d["terminal"]:
                y = d["rho"]
            else:
                nxt = np.zeros(len(phi))
                if d["ok"].any():
                    _, nxt[d["ok"]] = greedy_next(theta, d["next"])
                y = d["rho"] * nxt
            A += phi.T @ phi
            b += phi.T @ y
        theta[t] = np.linalg.solve(A, b)
    return theta


def train_backward(opponent=default_batch, n_rounds=6, games_per_round=8000,
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
        data, _ = collect(theta, epsilon, train_opp, games_per_round, seed + 1000 * r)
        buffer = (buffer + [data])[-buffer_rounds:]
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
def train_qlearning(opponent=default_batch, n_batches=60, games_per_batch=500,
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
            pred = phi @ theta
            if d["terminal"]:
                target = d["rho"]
            else:
                nxt = np.zeros(len(phi))
                if d["ok"].any():
                    _, nxt[d["ok"]] = greedy_next(theta, d["next"])
                target = d["rho"] * nxt
            err = np.clip(target - pred, -5, 5)        # clip rare huge ratios
            theta += alpha * phi.T @ err / len(phi)
        if (k + 1) % eval_every == 0:
            res = evaluate(theta, opponent, eval_games, seed=777)
            curve.append((k + 1, res))
            if verbose:
                print(f"Q-learning batch {k + 1}/{n_batches}: greedy mean B10 = "
                      f"{res['mean']:.1f} (median {res['median']:.1f})", flush=True)
    return theta, curve


def evaluate(theta, opponent, n_games=20000, seed=777):
    B1, B2 = play_chunked(greedy_batch(theta), opponent, n_games, seed=seed)
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
            super().__init__()
            self.common = nn.Linear(n_in, hidden)
            self.actor = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            h = F.relu(self.common(x))
            return self.actor(h), self.critic(h).squeeze(-1)

    return ActorCritic


def ac_policy(model, greedy=False):
    import torch

    def policy(st, rng):
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(ac_inputs(st)))
            if greedy:
                idx = logits.argmax(1).numpy()
            else:
                p = torch.softmax(logits, 1).numpy().astype(float)
                p /= p.sum(1, keepdims=True)
                c = p.cumsum(1)
                idx = (rng.uniform(size=(len(p), 1)) < c).argmax(1)
        policy.last_idx = idx
        return actions_from_index(st, idx)
    return policy


def train_ac(n_updates=12000, games=512, lr=1e-3, entropy=0.003, clip_return=1000.0,
             seed=0, opponent=default_batch, eval_every=1200, eval_games=10000, verbose=True):
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
        pol = ac_policy(model)
        idx_log = []
        _, _, log = play(pol, opponent, games, seed=seed * 100000 + k, record=True,
                         callback=lambda *a: idx_log.append(pol.last_idx.copy()))
        BT = log[-1]["B_next"]
        X, A, G = [], [], []
        for t in range(T):
            st = log[t]["state"]
            alive = st["B_self"] > 0
            X.append(ac_inputs({kk: v[alive] for kk, v in st.items()}))
            A.append(idx_log[t][alive])
            G.append(np.minimum(BT[alive] / st["B_self"][alive], clip_return))  # return per unit of wealth
        X = torch.from_numpy(np.vstack(X)); A = torch.from_numpy(np.concatenate(A)).long()
        G = torch.from_numpy(np.concatenate(G).astype(np.float32))
        logits, V = model(X)
        logp = torch.log_softmax(logits, 1)
        adv = (G - V).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)                  # standardise, as in PS6
        actor_loss = -(logp.gather(1, A[:, None]).squeeze(1) * adv).mean()
        critic_loss = F.huber_loss(V, G)
        ent = -(logp.exp() * logp).sum(1).mean()
        loss = actor_loss + critic_loss - entropy * ent
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (k + 1) % eval_every == 0:
            r = summarise(*play_chunked(ac_policy(model, greedy=True), opponent, eval_games, seed=777))
            curve.append((k + 1, r))
            if verbose:
                print(f"AC update {k + 1}: greedy mean B10 {r['mean']:.0f} ± {r['se']:.0f}, "
                      f"entropy {ent.item():.2f}", flush=True)
    return model, curve
