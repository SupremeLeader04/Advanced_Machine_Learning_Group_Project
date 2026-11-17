"""2.4 Refine your RL agents -- final agent, experiments, figures and tables.

    python refine_experiments.py
    python refine_experiments.py --evaluate

--evaluate runs the saved tournament policy against the default opponent in
the original Section 2.1 environment. It only requires NumPy.

RUN_EXPERIMENTS = False (default): trains the final agent against the default
opponent in the Section 2.1 environment and writes rlagent/agentE_theta.npy.

RUN_EXPERIMENTS = True: also runs the method comparisons in the Section 2.1
environment and writes updated tables and figures to refine_results/.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from default_opponent import default_opponent_policy
from environment import sim_game

RUN_EXPERIMENTS = False

OUT = Path("refine_results")
AGENT_FILE = Path("rlagent") / "agentE_theta.npy"

# ---------------------------------------------------------------------------
# Tuning parameters (Table table_2_4_tuning.csv)
# ---------------------------------------------------------------------------
FINAL_CFG = dict(n_rounds=6, games_per_round=6000, epsilon=0.1, ridge=1.0,
                 buffer_rounds=6, seed=0, eval_games=2000)
EXTRA_SEEDS = [1, 2, 3, 4]
AC_CFG = dict(n_updates=3000, games=256, lr=1e-3, entropy=0.003, clip_return=1000.0,
              seed=0, eval_every=300, eval_games=5000)   # 12,000 x 512 would take ~7 h on a CPU
QL_CFG = dict(n_batches=60, games_per_batch=1000, alpha0=0.05, eps0=0.3,
              eps_min=0.05, eval_every=10, eval_games=10000, seed=7)
POOL_CFG = dict(n_rounds=8, games_per_round=6000, epsilon=0.1, ridge=1.0,
                buffer_rounds=6, seed=50, eval_games=20000)
FP_CFG = dict(POOL_CFG, seed=90)
TEST_SEED, TEST_GAMES = 4242, 50000      # held-out evaluation vs default
ROBUST_SEED, ROBUST_GAMES = 31, 8000     # robustness matrix (per cell)

# REFINE_QUICK=1 shrinks everything for a fast smoke test (results not meaningful)
if os.environ.get("REFINE_QUICK"):
    for c in (FINAL_CFG, POOL_CFG, FP_CFG):
        c.update(n_rounds=2, games_per_round=400, eval_games=300)
    QL_CFG.update(n_batches=4, games_per_batch=200, eval_every=2, eval_games=300)
    AC_CFG.update(n_updates=20, eval_every=10, eval_games=300)
    EXTRA_SEEDS, TEST_GAMES, ROBUST_GAMES = [1], 1000, 300


def evaluate_saved_agent(n_games=TEST_GAMES, seed=TEST_SEED):
    """Evaluate the shipped tournament policy directly in the Section 2.1 game."""
    from rlagent.agentE import policy

    def agent(state):
        return list(policy(*state))

    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        budgets = np.empty((n_games, 2))
        for i in range(n_games):
            budgets[i] = sim_game(policy_1=agent, policy_2=default_opponent_policy)[-1, 1:3]
    finally:
        np.random.set_state(rng_state)

    own, opp = budgets.T
    print(f"{n_games:,} games against default in the Section 2.1 environment (seed {seed})")
    print(f"Mean terminal wealth: {own.mean():,.2f} (SE {own.std(ddof=1) / np.sqrt(n_games):,.2f})")
    print(f"Opponent mean: {opp.mean():,.2f}")
    print(f"Median terminal wealth: {np.median(own):,.2f}")
    print(f"Bankruptcy rate: {np.mean(own == 0):.2%}")
    print(f"Finished ahead of opponent: {np.mean(own > opp):.2%}")


def main():
    """Train the final agent and save its weights; optionally run all experiments."""
    from refine import train_backward
    t0 = time.time()
    theta, curve = train_backward(**FINAL_CFG)
    AGENT_FILE.parent.mkdir(exist_ok=True)
    np.save(AGENT_FILE, theta)
    print(f"Saved final agent to {AGENT_FILE} ({(time.time() - t0) / 60:.1f} min)")
    if RUN_EXPERIMENTS:
        run_experiments(theta, curve)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_experiments(theta_final, curve_final):
    """Run the method comparisons and write new tables and figures."""
    import refine
    from refine import (
        Mixture, ac_policy, greedy_batch, hold_batch, myopic_batch, rollouts,
        standard_pool, summarise, train_ac, train_backward, train_qlearning,
    )
    OUT.mkdir(exist_ok=True)

    # 1) stability across training seeds
    seed_rows = []
    for s in [FINAL_CFG["seed"]] + EXTRA_SEEDS:
        th = theta_final if s == FINAL_CFG["seed"] else \
            train_backward(**dict(FINAL_CFG, seed=s), verbose=False)[0]
        b1, b2 = rollouts(greedy_batch(th), default_opponent_policy, TEST_GAMES, seed=TEST_SEED)
        seed_rows.append(dict(training_seed=s, **summarise(b1, b2)))
        print(f"seed {s}: {seed_rows[-1]['mean']:.0f}", flush=True)
    _write_csv("table_2_4_seeds.csv", seed_rows)

    # 2) other methods and variants
    th_ql, curve_ql = train_qlearning(**QL_CFG)
    ac_model, curve_ac = train_ac(**AC_CFG)
    refine.FEATURE_SET = "plain"                       # ablation: group's 2.3 action terms
    th_plain, _ = train_backward(**FINAL_CFG, verbose=False)
    b1, b2 = rollouts(greedy_batch(th_plain), default_opponent_policy, TEST_GAMES, seed=TEST_SEED)
    refine.FEATURE_SET = "economic"
    ablation_row = dict(name="LSPI finite-horizon, 2.3 action features (ablation)",
                        **summarise(b1, b2), p_zero=float(np.mean(b1 == 0)),
                        p_above_100=float(np.mean(b1 > 100)))

    # opponents for pool training: the standard pool, any `extra` agents, and the
    # agent's own current policy (self-play), re-drawn every LSPI round
    def pool_factory(extra):
        def factory(theta):
            pool = list(standard_pool().values()) + extra
            w = [3, 1, 1, 1, 1, 1] + [2] * len(extra)
            if theta is not None:
                pool.append(greedy_batch(theta))
                w.append(2)
            return Mixture(pool, w)
        return factory

    th_pool, curve_pool = train_backward(opponent_factory=pool_factory([]), **POOL_CFG)
    th_fp, curve_fp = train_backward(
        opponent_factory=pool_factory([greedy_batch(theta_final), greedy_batch(th_pool)]),
        **FP_CFG)
    agents = {"LSPI finite-horizon (final)": greedy_batch(theta_final),
              "LSPI pool-trained": greedy_batch(th_pool),
              "LSPI fictitious play": greedy_batch(th_fp),
              "Q-learning (incremental)": greedy_batch(th_ql)}

    # 3) held-out performance against the default (same 50,000 test games for all)
    rows, budgets = [], {}
    bench = {"Default (firm 1 also default)": default_opponent_policy, "Hold cash": hold_batch,
             "One-step look-ahead (model-based)": myopic_batch("default")}
    others = {"Actor-critic NN (raw state)": ac_policy(ac_model, greedy=True)}
    for name, pol in {**bench, **agents, **others}.items():
        b1, b2 = rollouts(pol, default_opponent_policy, TEST_GAMES, seed=TEST_SEED)
        budgets[name] = b1
        rows.append(dict(name=name, **summarise(b1, b2), p_zero=float(np.mean(b1 == 0)),
                         p_above_100=float(np.mean(b1 > 100))))
        print(f"{name:36s} {rows[-1]['mean']:9.1f} ± {rows[-1]['se']:6.1f}", flush=True)
    rows += [ablation_row] + evaluate_23_baselines()
    _write_csv("table_2_4_performance_vs_default.csv", rows)

    # 4) robustness matrix: every agent against every opponent type
    opponents = {**standard_pool(), **agents}
    robust = []
    for an, ap in agents.items():
        for on, op in opponents.items():
            b1, b2 = rollouts(ap, op, ROBUST_GAMES, seed=ROBUST_SEED)
            robust.append(dict(agent=an, opponent=on, **summarise(b1, b2)))
        print(f"robustness done: {an}", flush=True)
    _write_csv("table_2_4_robustness.csv", robust)

    write_tuning_table()
    curves = {"final": curve_final, "pool": curve_pool, "fp": curve_fp, "ql": curve_ql,
              "ac": curve_ac}
    make_figures(rows, curves, budgets, robust, theta_final)
    print(f"All results written to {OUT}/")


def evaluate_23_baselines(n_games=2000):
    """Train and evaluate the group's 2.3 MC and SARSA agents (control.py)."""
    import control
    rows = []
    for method, alpha in (("MC", 0.001), ("SARSA", 0.005)):
        for feats in ("baseline", "quadratic"):
            n_ep = 3000 if not os.environ.get("REFINE_QUICK") else 50
            theta, _, _ = control.train_control(method, feats, n_ep, 123, n_ep, 20, alpha)
            r = control.evaluate_policy(theta, feats, n_games, seed=TEST_SEED)
            rows.append(dict(name=f"2.3 {method} ({feats} features)", mean=r["mean"],
                             se=r["se"], median=r["median"], win_rate=r["win_rate"],
                             opp_mean=float(np.mean(r["opponent"])),
                             p_zero=float(np.mean(r["own"] == 0)),
                             p_above_100=float(np.mean(r["own"] > 100))))
    return rows


def _write_csv(name, rows):
    """Write a list of dicts to refine_results/<name> (columns = union of keys)."""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_tuning_table():
    """Write the table of tuning choices (table_2_4_tuning.csv)."""
    from refine import H_NAMES, MAX_EFFECTIVE_SPEED, SPEED_FRACS, TRADE_FRACS
    rows = [
        ("speed grid u = δ²/B", ", ".join(f"{x:g}" for x in SPEED_FRACS),
         "finer at small spends: δ=1 costs only 1% of B=100 but changes P(win) a lot"),
        ("maximum effective speed for our agent", f"{MAX_EFFECTIVE_SPEED:g}",
         "prevents overflow when the supplied simulator computes exp(2 × effective speed)"),
        ("trade grid v = |x|/W", ", ".join(f"{x:g}" for x in TRADE_FRACS), "as in 2.3; direction = sign(g)"),
        ("action terms h(s,a)", " | ".join(H_NAMES), "interacted with the 50 state indicators of 2.3.1"),
        ("Q scaling", "Q = B_t · θ_t'φ", "value roughly proportional to wealth; target ratio ρ = B_{t+1}/B_t"),
        ("LSPI rounds", FINAL_CFG["n_rounds"], "same number of rounds as the exploratory run"),
        ("games per round", FINAL_CFG["games_per_round"], "about 60k transitions per round"),
        ("behaviour ε (LSPI)", FINAL_CFG["epsilon"], "0.1 and 0.2 gave the same final value"),
        ("replay buffer", f"{FINAL_CFG['buffer_rounds']} rounds (all data)", "off-policy: old data stays valid"),
        ("ridge penalty", FINAL_CFG["ridge"], "stabilises OLS for rarely visited bins"),
        ("Q-learning α_k", f"{QL_CFG['alpha0']}/(1+k/20)^0.6", "decreasing step size, per batch"),
        ("Q-learning ε_k", f"max({QL_CFG['eps_min']}, {QL_CFG['eps0']}/(1+k/10))", "GLIE-style decay"),
        ("Q-learning TD-error clip", "±5", "limits rare huge ratios ρ"),
        ("actor-critic network", "59 inputs → 128 ReLU (shared) → actor 176 logits, critic 1",
         "as in mini-problem set 6; inputs: 9 scaled state variables + 50 indicators"),
        ("actor-critic training", f"{AC_CFG['n_updates']} updates × {AC_CFG['games']} games, Adam "
         f"lr {AC_CFG['lr']}/(1+5k/K), entropy {AC_CFG['entropy']}",
         "decreasing step size; sized to run in about an hour on a CPU"),
        ("actor-critic return", f"ρ = B_T/B_t clipped at {AC_CFG['clip_return']:g}; standardised advantage; Huber critic",
         "per-unit-wealth return as for LSPI; clipping limits rare huge returns"),
        ("pool training weights", "default 3, hold 1, all-in 1, look-ahead 1, const-speed-2 1, "
         "const-speed-5 1, self 2", "robustness for the tournament"),
    ]
    _write_csv("table_2_4_tuning.csv", [dict(parameter=a, value=b, reason=c) for a, b, c in rows])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(perf, curves, budgets, robust, theta_final):
    """Draw the five 2.4 figures from the experiment results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, grey = "#1f77b4", "#7f7f7f"
    ref = {r["name"]: r["mean"] for r in perf}

    # Figure: learning curves (greedy-policy value during training)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for key, lab, col in [("final", "LSPI finite-horizon (vs default) – final", blue),
                          ("pool", "LSPI pool-trained", "#ff7f0e"),
                          ("fp", "LSPI fictitious play", "#2ca02c")]:
        x = np.arange(1, len(curves[key]) + 1)
        m = np.array([r["mean"] for r in curves[key]])
        se = np.array([r["se"] for r in curves[key]])
        axes[0].plot(x, m, marker="o", color=col, label=lab)
        axes[0].fill_between(x, m - 1.96 * se, m + 1.96 * se, color=col, alpha=.15)
    axes[1].plot([k for k, _ in curves["ql"]], [r["mean"] for _, r in curves["ql"]],
                 marker="o", color="#9467bd", label="Q-learning (incremental)")
    axes[2].plot([k for k, _ in curves["ac"]], [r["mean"] for _, r in curves["ac"]],
                 marker="o", color="#8c564b", label="Actor-critic NN (greedy policy)")
    for ax in axes:
        ax.axhline(ref["One-step look-ahead (model-based)"], ls="--", color="k", lw=1,
                   label="One-step look-ahead (model-based)")
        ax.axhline(100, ls=":", color=grey, label="Hold cash (= 100)")
        ax.axhline(ref["2.3 MC (baseline features)"], ls="-.", color="brown", lw=1, label="2.3 MC control")
        ax.set_yscale("log"); ax.set_ylim(50, 12000)
        ax.grid(alpha=.25, which="both"); ax.legend(fontsize=8, loc="center right")
    axes[0].set(xlabel="LSPI round (6,000 new ε-greedy games per round)",
                ylabel="Greedy mean terminal budget vs default", title="Batch control: finite-horizon LSPI")
    axes[1].set(xlabel="Q-learning batch (1,000 games per batch)", title="Incremental Q-learning")
    axes[2].set(xlabel=f"Actor-critic update ({AC_CFG['games']} games per update)", title="Actor-critic (neural network)")
    _save(fig, "fig_2_4_learning_curves.png")

    # Figure: held-out performance of every policy against the default
    p = sorted(perf, key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = [r["name"] for r in p]
    m = np.array([r["mean"] for r in p]); se = np.array([r["se"] for r in p])
    cols = [blue if "final" in n else grey if any(k in n for k in ("2.3", "Hold", "Default"))
            else "#ff7f0e" for n in names]
    ax.barh(names, np.maximum(m, .5), xerr=1.96 * se, color=cols)
    ax.set_xscale("log"); ax.axvline(100, ls=":", color="k"); ax.set_xlim(.3, m.max() * 30)
    for i, r in enumerate(p):
        ax.text(max(r["mean"], .5) * 1.15, i, f"mean {r['mean']:,.0f} | median {r['median']:,.0f}",
                va="center", fontsize=8)
    ax.set(xlabel="Mean terminal budget B_10 against the default opponent (log scale, 95% CI)",
           title="Held-out performance against the default strategy")
    _save(fig, "fig_2_4_performance_vs_default.png")

    # Figure: what the final policy does in individual games
    from refine import U_GRID, V_GRID, greedy_batch, rollouts
    _, _, log = rollouts(greedy_batch(theta_final), default_opponent_policy, min(20000, TEST_GAMES),
                         seed=11, record=True)
    rec = {k: [] for k in ("t", "u", "v", "delta", "sd", "g", "R", "p")}
    for t in range(10):
        s = log[t]["state"]; a = s["B_self"] > 0
        idx = log[t]["idx"][a]
        for k, v in (("t", np.full(a.sum(), t)), ("u", U_GRID[idx]), ("v", V_GRID[idx]),
                     ("delta", log[t]["delta"][a]), ("sd", (s["S_self"] - s["S_opp"])[a]),
                     ("g", np.abs(s["g"][a])), ("R", s["R"][a]), ("p", log[t]["win"][a])):
            rec[k].append(v)
    r = {k: np.concatenate(v) for k, v in rec.items()}
    ts = np.arange(10)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    for key, mk, lab in [("u", "o", "speed spend δ²/B"), ("v", "s", "|x| / W (share of wealth bet)"),
                         ("p", "^", "Speed race win rate")]:
        ax.plot(ts, [r[key][r["t"] == t].mean() for t in ts], marker=mk, label=lab)
    ax.set(xlabel="Period t", ylim=(0, 1.05), title="Policy over time (firms not bankrupt)"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(ts, [r["delta"][r["t"] == t].mean() for t in ts], marker="o", label="speed investment δ")
    ax.plot(ts, [r["sd"][r["t"] == t].mean() for t in ts], marker="s", label="speed lead S_self − S_opp")
    ax.set(xlabel="Period t", title="Speed: invest early, then maintain the lead"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    edges = [-3, -1, 0, 1, 2, 3, 5]; lab = ["<-3", "-3..-1", "-1..0", "0..1", "1..2", "2..3", "3..5", ">5"]
    b = np.searchsorted(edges, r["sd"], "right"); late = r["t"] >= 1
    ax.bar(lab, [r["u"][(b == i) & late].mean() if np.any((b == i) & late) else 0 for i in range(len(lab))])
    ax.set(xlabel="Speed lead at start of period (t ≥ 1)", ylabel="mean δ²/B", title="Speed spend vs current speed lead")
    ax = axes[1, 1]
    ge = [0, .25, .5, 1, 1.5, 2, 3, 9]; gb = np.searchsorted(ge[1:-1], r["g"], "right")
    for lo, hi, mk in [(0, .5, "o"), (.5, 1.01, "s")]:
        msk = (r["R"] >= lo) & (r["R"] < hi)
        ax.plot(range(len(ge) - 1), [r["v"][msk & (gb == i)].mean() if np.any(msk & (gb == i)) else np.nan
                                     for i in range(len(ge) - 1)], marker=mk, label=f"R in [{lo}, {min(hi, 1)}]")
    ax.set_xticks(range(len(ge) - 1), [f"{ge[i]}–{ge[i + 1]}" for i in range(len(ge) - 1)], rotation=30)
    ax.set(xlabel="|g|", ylabel="mean |x| / W", ylim=(0, 1.05), title="Bet size vs signal: (almost) always all-in")
    ax.legend(fontsize=8)
    for a in axes.ravel():
        a.grid(alpha=.25)
    _save(fig, "fig_2_4_policy.png")

    # Figure: distribution of terminal wealth and of tournament-sized averages
    bf, bl = budgets["LSPI finite-horizon (final)"], budgets["One-step look-ahead (model-based)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arr, lab, col in [(bf, "LSPI final", blue), (bl, "One-step look-ahead", "k")]:
        s = np.sort(arr)
        axes[0].plot(np.maximum(s, .1), np.arange(1, len(s) + 1) / len(s), color=col, label=lab)
    axes[0].axvline(100, ls=":", color=grey, label="Hold cash")
    axes[0].set(xscale="log", xlabel="Terminal budget B_10 (0 shown at 0.1)", ylabel="Empirical CDF",
                title="Terminal wealth vs default: mostly bust, rare huge wins"); axes[0].legend(fontsize=8)
    rng = np.random.default_rng(0)
    for n, col in [(10, "#aec7e8"), (90, blue)]:
        mm = np.array([rng.choice(bf, n).mean() for _ in range(20000)])
        axes[1].hist(np.log10(np.maximum(mm, 1)), bins=60, alpha=.7, color=col,
                     label=f"average of {n} games: P(<100) = {np.mean(mm < 100):.3f}")
    axes[1].axvline(2, ls=":", color="k")
    axes[1].set(xlabel="log10(average terminal budget)", ylabel="count (bootstrap)",
                title="Tournament-sized samples (vs default)"); axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=.25)
    _save(fig, "fig_2_4_wealth_distribution.png")

    # Figure: robustness heat map (agents x opponents)
    rows_ = list(dict.fromkeys(r["agent"] for r in robust))
    cols_ = list(dict.fromkeys(r["opponent"] for r in robust))
    M = np.array([[next(x["mean"] for x in robust if x["agent"] == a and x["opponent"] == o)
                   for o in cols_] for a in rows_])
    fig, ax = plt.subplots(figsize=(13, 3.8))
    im = ax.imshow(np.log10(np.maximum(M, 1)), cmap="viridis", aspect="auto")
    for i in range(len(rows_)):
        for j in range(len(cols_)):
            ax.text(j, i, f"{M[i, j]:,.0f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xticks(range(len(cols_)), cols_, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows_)), [f"{a}  (avg {M[i].mean():,.0f})" for i, a in enumerate(rows_)], fontsize=8)
    ax.set_title(f"Mean terminal budget of row agent against column opponent ({ROBUST_GAMES:,} games each)")
    fig.colorbar(im, label="log10 mean B_10")
    _save(fig, "fig_2_4_robustness.png")


def _save(fig, name):
    """Save a figure to refine_results/ and close it."""
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce or evaluate the Group E agent.")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate the saved agent in the Section 2.1 simulator")
    if parser.parse_args().evaluate:
        evaluate_saved_agent()
    else:
        main()
