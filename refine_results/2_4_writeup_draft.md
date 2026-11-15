# 2.4 Refine Your RL Agents (draft text for the write-up)

All numbers come from `refine_experiments.py` (fixed seeds). "Mean B10" is the mean terminal budget of our firm against the default opponent over 50,000 held-out games (seed 4242), unless stated otherwise. Figures and tables are in `refine_results/`.

## Which methods from class have the most potential, and why

Our 2.3 agents did not beat the default strategy in a useful way:
- MC control converged to "hold cash" (mean B10 ≈ 101–105).
- SARSA went bankrupt (mean B10 ≈ 0–5) (Table `table_2_4_performance_vs_default.csv`).

We trace this to three problems:

1. **On-policy learning with ε-soft exploration.** SARSA and on-policy MC evaluate the *exploring* policy. In this game one random all-in bet or random 100% speed spend can wipe out the budget, so the value of every action is learned under a policy that often destroys wealth.
2. **Scale.** Terminal wealth is heavy-tailed (from 0 to over 10⁵), and the Q-function has to represent values proportional to the current budget. Bin indicators interacted with only `u` and `v` cannot represent this.
3. **Action features with no economic content.** Q was linear in the speed fraction `u` and the trade fraction `v`. The real effect of speed works through the softmax win probability (very nonlinear in δ). The effect of trading works through the convex payoff `max(0, W + xΔ)`.

Among the methods from class we have not used yet:

| Method (lecture) | Potential | Our assessment |
|---|---|---|
| Q-learning, off-policy TD control (L4) | learns the greedy policy while exploring | high: fixes problem 1 |
| LSTDQ / LSPI, batch off-policy control (L5) | reuses all simulated data, solves for weights in closed form | high: stable and data-efficient, since the simulator is cheap |
| Off-policy MC with importance sampling (L4) | — | low: importance-weight products over 10 steps have huge variance |
| DQN: neural network with experience replay and fixed targets (L5, mini-PS 5) | flexible nonlinear Q | medium: our batch LSPI already uses experience replay; a neural Q is costly to tune with heavy-tailed returns |
| Actor-critic with a neural network (L6, mini-PS 6) | learns the policy directly from the raw state, with no hand-built features | high: tests whether our features matter; the risk is high gradient variance from heavy-tailed returns |
| Dynamic programming / backward induction (L2) | exact for a known model and a finite horizon | high as a benchmark and as a structural idea; we use it in the "finite-horizon LSPI" below |

We implemented **Q-learning**, **LSPI** and an **actor-critic neural network**. We kept the group's 50 state indicators from 2.3.1 and changed three things:
- the **Q architecture**,
- the **action features**,
- the **finite-horizon structure**.

## What we implemented

**Wealth-scaled linear Q-function.** Expected terminal wealth is roughly proportional to current wealth, so we write

  Q(s, a) = B_t · θ_tᵀ φ(s, a).

This is still linear value function approximation. Only the features change, to B_t·φ. Dividing the Bellman equation by B_t gives a TD target per unit of wealth, θ_tᵀφ(s,a) ≈ ρ · max_a′ θ_{t+1}ᵀφ(s′,a′) with ρ = B_{t+1}/B_t, and θ_9ᵀφ ≈ ρ in the last period. Bankrupt states are absorbing, with value 0.

**Features φ(s,a) = z(s) ⊗ h(s,a)** (300 weights per period):
- z(s) is the group's 50 binary indicators from 2.3.1.
- h(s,a) contains six action terms with economic meaning, each multiplied by the kept wealth share (1−u):
  - kept wealth (1−u);
  - P_win(default opp) × gain(v);
  - P_win(passive opp) × gain(v);
  - P_win(default opp);
  - P_win(passive opp);
  - v × P_win(default opp).

Here gain(v) = E[max(0, 1 + v·sign(g)Δ)] − 1 is the closed-form expected trading return given (g, R). P_win(·) is the softmax win probability given our δ, S_self and S_opp, assuming either a default-like opponent (δ_opp = Z·√B_opp) or a passive one (δ_opp = 0). These terms use only the rules of the game and the current state. They are features, not a fixed policy: the learned weights decide how much each term matters in each state.

**Action grid.**
- The speed spend u = δ²/B is finer at small values: {0, .25%, .5%, 1%, 2%, 3%, 5%, 7.5%, 10%, 15%, 20%, 30%, 40%, 50%, 70%, 100%}. At B = 100, δ = 1 costs only 1% of the budget but moves the win probability a lot.
- The trade share v = |x|/W is in {0, 0.1, …, 1}, with direction sign(g), as in 2.3.

**(a) Incremental Q-learning (L4).** This is semi-gradient Q-learning on the scaled Q, with weights shared across periods.
- Data: 60 batches of 1,000 ε-greedy games each.
- Step size: α_k = 0.05/(1+k/20)^0.6.
- Exploration: ε_k = max(0.05, 0.3/(1+k/10)).
- Updates sweep backwards through each batch, and TD errors are clipped at ±5.

**(b) LSPI with LSTDQ (L5), first attempt with shared weights across periods.** The policy oscillated between rounds (greedy mean B10 jumped between ≈20 and ≈1,800). This is a known weakness of LSPI under approximation error: argmax policies switch, and heavy-tailed ratios ρ make the fitted Q noisy.

**(c) Finite-horizon LSPI (final agent).** We give each period its own weights θ_t. The LSTDQ system Σφ(φ − ρφ′_π)ᵀθ = Σφρ·1[t=9] then becomes block-triangular, because period t only bootstraps from period t+1. Its solution is exactly **backward induction**:
1. OLS of ρ on φ for t = 9.
2. For t = 8, OLS of ρ·max_a′ Q_9(s′,a′).
3. Continue back to t = 0.

In other words, this is batch Q-learning, or approximate DP, using the simulated data. Each round:
- collects 6,000 new ε-greedy games (ε = 0.1) with the current greedy policy;
- adds them to a replay buffer that keeps all past rounds (the method is off-policy, so old data stays valid);
- refits all periods backwards with ridge penalty 1.

We ran 6 rounds; the LSPI value curve is flat after round 4 (Figure `fig_2_4_learning_curves.png`). **Stability across training seeds:** we retrained the agent with 4 further seeds (`table_2_4_seeds.csv`). Evaluated on the same 50,000 held-out games, the five agents score 5,593, 5,625, 5,638, 5,642 and 5,616 (mean 5,623, SD 20 across seeds). The training randomness therefore barely matters; the ±250 standard errors come from evaluation noise, because terminal wealth is so heavy-tailed. Tuning choices are listed in `table_2_4_tuning.csv`. Training takes about 5 minutes on one CPU core.

**(d) Actor-critic with a neural network (L6, set up as in mini-problem set 6).** The network has 59 inputs (9 scaled state variables plus the 50 indicators), a shared 128-unit ReLU layer, an actor head (softmax preferences over the same 176 actions) and a critic head.
- Return: the Monte-Carlo return per unit of wealth, ρ = B_T/B_t, clipped at 1,000.
- Losses: the advantage is ρ − V(s), standardised as in PS6; the critic uses a Huber loss; there is a small entropy bonus (0.003).
- Training: Adam with a decreasing step size, 12,000 updates of 512 games each (about 6 minutes on one CPU).
- Unlike LSPI, it sees **only the raw state**, with no rules-based action features.
- Longer training and larger batches (2,048 games) gave the same value, about 4,570 (Figure `fig_2_4_learning_curves.png`, right panel).

**(e) Feature ablation.** We reran the final method (wealth-scaled, finite-horizon LSPI) with the group's 2.3 quadratic action terms (1, u, v, u², v², uv) in place of the rules-based terms. This separates the contribution of the method from that of the features.

## Performance against the default strategy

(Figure `fig_2_4_performance_vs_default.png`, Table `table_2_4_performance_vs_default.csv`; 50,000 held-out games.)

| Policy | mean B10 (± s.e.) | median | P(B10 = 0) | P(B10 > 100) |
|---|---|---|---|---|
| **Finite-horizon LSPI (final)** | **5,593 ± 253** | 0 | 63% | 22% |
| LSPI fictitious play | 5,049 ± 236 | 0 | 58% | 22% |
| LSPI pool-trained | 4,937 ± 232 | 0 | 58% | 22% |
| Q-learning (incremental) | 4,591 ± 241 | 0 | 79% | 19% |
| Actor-critic NN (raw state only) | 4,588 ± 200 | 0 | 56% | 22% |
| LSPI finite-horizon with the 2.3 action features (ablation) | 3,267 ± 152 | 0 | 53% | 20% |
| One-step look-ahead (model-based benchmark) | 2,842 ± 210 | 95 | 12% | 9% |
| 2.3 MC (quadratic / baseline features) | 105 / 101 | 100 | 1% | 2% |
| Hold cash | 100 | 100 | 0% | 0% |
| Default vs default | 11 | 0.8 | 1% | 2% |
| 2.3 SARSA (quadratic / baseline features) | 5 / 0 | 0 | 76–81% | ≈0% |

- Finite-horizon LSPI is about 55 times better than holding cash and about 2× the one-step look-ahead benchmark, which already knows the model.
- Incremental Q-learning reaches about 80% of the LSPI value with the same features. It is noisier and uses data less efficiently.
- **Where the gain comes from** (ablation):
  - Keeping the group's own 2.3 action features but switching to the wealth-scaled, finite-horizon LSPI raises mean B10 from about 100 to 3,267. Most of the improvement therefore comes from the method.
  - The rules-based action features add a further 70% (to 5,593).
  - The actor-critic, which sees only the raw state, reaches 4,588. A flexible learner without any hand-built features gets about 80% of the way, which suggests the features help learning but do not dictate the policy.

## The policy we end up with

(Figure `fig_2_4_policy.png`.)

- **Front-loaded speed race.** In period 0 the agent always spends about 29% of its budget on speed (δ ≈ 5.4). Spending then falls steadily, to about 0.3% by period 9. Speed capital decays slowly (by 3.3% a period on average), while the default opponent burns about a third of its budget every period. An early lead is therefore cheap to keep: the agent's win probability rises from 54% at t = 0 to about 82% at t = 4–5.
- **Speed responds to the race position.** The agent spends most when it is slightly behind (lead between −3 and 0), where an extra unit of speed changes the outcome most. It spends little when far ahead (no need) or far behind (too expensive to catch up).
- **Almost always all-in in the direction of the signal.** About 93% of post-speed wealth is bet, for both weak and strong signals and for low and high R. This is optimal for expected wealth because of **limited liability**: B_{t+1} = max(0, W + xΔ) makes the payoff convex in x (like a call option on Δ). So E[B_{t+1}] increases in |x| even when the signal is weak, and a risk-neutral firm wants the largest position.
- **Risk profile** (Figure `fig_2_4_wealth_distribution.png`). The firm goes bankrupt in 63% of games, but 15% of games end above 1,000 and the 99th percentile is about 107,000. The objective E[B_T] rewards exactly this. For tournament-sized samples (averages over 90 games), a bootstrap gives P(average < 100) ≈ 0.005% against the default. Over 10 games the probability is 18%, so single match-ups are noisy.

## Robustness: other opponents and the tournament

(Figure `fig_2_4_robustness.png`, Table `table_2_4_robustness.csv`; 8,000 games per cell.)

The tournament opponents will not play the default, so we also trained two variants:
- **Pool-trained LSPI:** opponents drawn from {default ×3, hold cash, all-in without speed, one-step look-ahead, constant δ = 2, constant δ = 5, current self ×2}.
- **Fictitious play:** the same pool plus earlier LSPI agents.

All three LSPI agents earn far more than 100 against every opponent we tried, between 2,778 and 12,249. Averaged over all ten opponents they are tied: 6,992, 7,092 and 7,029. The pool-trained agent exploits passive opponents better (12,249 vs 10,513 against hold cash). The default-trained agent beats it head to head: it earns 5,152 against the pool-trained agent, while the pool-trained agent earns 2,778 against it. It is also slightly ahead against the fictitious-play agent (4,162 vs 3,570). We submit the **default-trained finite-horizon LSPI** agent for four reasons:
1. It is the best against the default strategy.
2. It is strongest against other learned speed-racing agents, which is what classmates who trained against the default are likely to submit.
3. It is no worse on average across opponents.
4. It is the simplest to reproduce.

## Limitations and possible extensions

- The action features assume the opponent's speed spend looks like the default's or like zero. A very different opponent (for example, one that always pays for maximum speed) is only covered through the state indicators.
- Heavy tails make evaluation expensive: standard errors of ±250 need 50,000 games.
- Natural next steps: a Gaussian actor-critic over continuous (u, v) initialised from this policy, and a neural-network Q (DQN) on the same scaled targets.

## Reproducing (for the PDF readme)

```
pip install numpy scipy matplotlib
python refine_experiments.py      # trains the final agent -> rlagent/agentE_theta.npy (~5 min CPU)
```

Set `RUN_EXPERIMENTS = True` at the top of `refine_experiments.py` to regenerate every figure and table of 2.4 in `refine_results/` (about 80 min on a 2-core CPU; the actor-critic also needs `pip install torch`). All seeds are fixed, so the numbers are reproduced exactly.

The 2.4 files are:
- `refine_env.py`: a vectorised copy of `environment.sim_game` (checked against it), the closed-form trade gain, the one-step look-ahead benchmark and the opponent pool.
- `refine.py`: the features, Q-learning, finite-horizon LSPI and the actor-critic.
- `refine_experiments.py`: the entry point.
- `rlagent/agentE.py` and `rlagent/agentE_theta.npy`: the tournament agent. Like the mini-problem sets, its learned data is a `.npy` file loaded with numpy. It needs only numpy and calls `policy(t, B_self, B_opp, S_self, S_opp, g, R) -> (delta, x)`.
