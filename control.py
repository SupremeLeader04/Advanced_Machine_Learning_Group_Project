# 2.3.3: On-policy Monte Carlo and SARSA control

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from default_opponent import default_opponent_policy
from environment import sim_game
from features import feasible_actions, state_action_features, state_features

FEATURE_SIZES = {"baseline": 150, "quadratic": 300}


def learning_rate(
    episode: int,
    alpha_0: float,
    decay_scale: float = 750.0,
    exponent: float = 0.6
) -> float:
    """Decreasing semi-gradient step size (one value per training episode)."""

    return alpha_0 / (1.0 + episode / decay_scale) ** exponent


def exploration_rate(
    episode: int,
    epsilon_0: float = 0.30,
    epsilon_min: float = 0.05,
    decay_scale: float = 750.0,
    exponent: float = 0.8
) -> float:
    """Decay exploration while retaining a small probability of random actions."""

    return max(
        epsilon_min,
        epsilon_0 / (1.0 + episode / decay_scale) ** exponent
    )


def action_fractions(
    state: list,
    actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised speed spending u and absolute trading exposure v."""

    budget = float(state[1])
    actions = np.asarray(actions, dtype=float).reshape(-1, 2)

    speed_cost = actions[:, 0] ** 2
    wealth = np.maximum(0.0, budget - speed_cost)

    u = speed_cost / budget if budget > 0 else np.zeros(len(actions))

    v = np.divide(
        np.abs(actions[:, 1]),
        wealth,
        out=np.zeros(len(actions)),
        where=wealth > 0
    )

    return u, v


def encode_action(
    state: list,
    action: list | tuple,
    features: str
) -> np.ndarray:
    """Features for the chosen action."""

    if features == "baseline":
        return state_action_features(state, action).astype(float)

    if features != "quadratic":
        raise ValueError("features must be 'baseline' or 'quadratic'.")

    z = state_features(state).astype(float)

    u, v = action_fractions(
        state,
        np.asarray([action], dtype=float)
    )

    u, v = float(u[0]), float(v[0])

    # Retains all original features and adds curvature and an action interaction.
    return np.concatenate([
        z,
        u * z,
        v * z,
        u * u * z,
        v * v * z,
        u * v * z
    ])


def action_values(
    state: list,
    actions: list,
    theta: np.ndarray,
    features: str
) -> np.ndarray:
    """
    Score all feasible actions without building 111 full feature vectors.

    Q(s,a) = theta.T @ phi(s,a); coefficients depend on the current state.
    This vectorised calculation is identical to scoring each action separately.
    """

    if features not in FEATURE_SIZES:
        raise ValueError("Unknown feature representation.")

    if theta.shape != (FEATURE_SIZES[features],):
        raise ValueError("Incorrect parameter vector length.")

    z = state_features(state).astype(float)

    coefficients = theta.reshape(-1, 50) @ z

    u, v = action_fractions(
        state,
        np.asarray(actions, dtype=float)
    )

    q = coefficients[0] + coefficients[1] * u + coefficients[2] * v

    if features == "quadratic":
        q = (
            q
            + coefficients[3] * u * u
            + coefficients[4] * v * v
            + coefficients[5] * u * v
        )

    if not np.all(np.isfinite(q)):
        raise FloatingPointError(
            "Nonfinite Q-values: reduce learning rate or inspect returns."
        )

    return q


def choose_action(
    state: list,
    theta: np.ndarray,
    epsilon: float,
    features: str,
    random_ties: bool = True
) -> tuple[list, np.ndarray]:
    """
    Choose a feasible action epsilon-greedily and return its feature vector.

    Random tie-breaking during training prevents an initially all-zero Q
    function from always selecting the first grid action.
    """

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    actions = feasible_actions(state)

    if epsilon > 0 and np.random.random() < epsilon:
        index = int(np.random.randint(len(actions)))

    else:
        q = action_values(state, actions, theta, features)
        best = np.flatnonzero(q == np.max(q))

        index = (
            int(np.random.choice(best))
            if random_ties
            else int(best[0])
        )

    action = list(actions[index])

    return action, encode_action(state, action, features)


def mc_update(
    theta: np.ndarray,
    trajectory: list[np.ndarray],
    terminal_budget: float,
    alpha: float
) -> None:
    """Every-visit on-policy Monte Carlo control."""

    for phi in trajectory:
        error = terminal_budget - float(theta @ phi)

        theta += alpha * error * phi


def sarsa_update(
    theta: np.ndarray,
    phi: np.ndarray,
    next_phi: np.ndarray | None,
    alpha: float,
    terminal_budget: float | None = None
) -> None:
    """On-policy one-step TD update with the *actually chosen* next action."""

    prediction = float(theta @ phi)

    if terminal_budget is not None:
        target = float(terminal_budget)

    else:
        if next_phi is None:
            raise ValueError(
                "A next action feature vector is required before termination."
            )

        target = float(theta @ next_phi)

    theta += alpha * (target - prediction) * phi


class MonteCarloAgent:
    """Choose actions epsilon-greedily; store a full episode for MC updates."""

    def __init__(
        self,
        theta: np.ndarray,
        epsilon: float,
        features: str
    ):
        self.theta = theta
        self.epsilon = epsilon
        self.features = features

        self.trajectory: list[np.ndarray] = []

    def __call__(self, state: list) -> list:
        action, phi = choose_action(
            state,
            self.theta,
            self.epsilon,
            self.features,
            random_ties=True
        )

        self.trajectory.append(phi)

        return action


class SarsaAgent:
    """Update when the simulator reveals the next state and action."""

    def __init__(
        self,
        theta: np.ndarray,
        epsilon: float,
        alpha: float,
        features: str
    ):
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.features = features

        self.previous_phi: np.ndarray | None = None
        self.decisions = 0

    def __call__(self, state: list) -> list:

        # Choose A_(t+1) with the current policy before updating (S_t, A_t).
        action, phi = choose_action(
            state,
            self.theta,
            self.epsilon,
            self.features,
            random_ties=True
        )

        if self.previous_phi is not None:
            sarsa_update(
                self.theta,
                self.previous_phi,
                phi,
                self.alpha
            )

        self.previous_phi = phi
        self.decisions += 1

        return action

    def finish_episode(self, terminal_budget: float) -> None:
        """Use the observed final budget for the last transition's TD target."""

        if self.previous_phi is not None:
            sarsa_update(
                self.theta,
                self.previous_phi,
                None,
                self.alpha,
                terminal_budget=terminal_budget
            )


class GreedyAgent:
    """Evaluation policy: epsilon=0, fixed weights, no learning or RNG draws."""

    def __init__(
        self,
        theta: np.ndarray,
        features: str
    ):
        self.theta = theta
        self.features = features

    def __call__(self, state: list) -> list:
        action, _ = choose_action(
            state,
            self.theta,
            epsilon=0.0,
            features=self.features,
            random_ties=False
        )

        return action


def evaluate_policy(
    theta: np.ndarray | None,
    features: str,
    n_games: int,
    seed: int
) -> dict:
    """
    Evaluate terminal budgets against default; None means default-v-default.

    The random-number state is restored afterwards so evaluation never
    changes the sequence of subsequent training episodes.
    """

    saved_rng = np.random.get_state()

    try:
        np.random.seed(seed)

        firm_1 = (
            default_opponent_policy
            if theta is None
            else GreedyAgent(theta, features)
        )

        own, opponent = [], []

        for _ in range(n_games):
            history = sim_game(
                n=10,
                policy_1=firm_1,
                policy_2=default_opponent_policy
            )

            own.append(float(history[-1, 1]))
            opponent.append(float(history[-1, 2]))

    finally:
        np.random.set_state(saved_rng)

    own = np.asarray(own)
    opponent = np.asarray(opponent)

    return {
        "mean": float(np.mean(own)),
        "median": float(np.median(own)),
        "se": (
            float(np.std(own, ddof=1) / np.sqrt(n_games))
            if n_games > 1
            else 0.0
        ),
        "win_rate": float(np.mean(own > opponent)),
        "mean_difference": float(np.mean(own - opponent)),
        "own": own,
        "opponent": opponent,
    }


def train_control(
    method: str,
    features: str,
    n_episodes: int,
    seed: int,
    evaluate_every: int,
    n_eval_games: int,
    alpha_0: float,
    epsilon_0: float = 0.30,
    reward_shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains an MC or SARSA agent against the default opponent.

    reward_shift=0 uses terminal wealth, while
    reward_shift=100 uses profit relative to the initial budget.
    """

    if method not in ("MC", "SARSA"):
        raise ValueError("method must be 'MC' or 'SARSA'.")

    if features not in FEATURE_SIZES:
        raise ValueError("Unknown feature representation.")

    np.random.seed(seed)

    theta = np.zeros(FEATURE_SIZES[features], dtype=float)

    checkpoints = [0]
    curve = [
        evaluate_policy(theta, features, n_eval_games, seed=9000)["mean"]
    ]

    for episode in range(1, n_episodes + 1):
        alpha = learning_rate(episode, alpha_0)
        epsilon = exploration_rate(episode, epsilon_0=epsilon_0)

        if method == "MC":
            agent = MonteCarloAgent(theta, epsilon, features)

        else:
            agent = SarsaAgent(theta, epsilon, alpha, features)

        history = sim_game(
            n=10,
            policy_1=agent,
            policy_2=default_opponent_policy
        )

        terminal_budget = float(history[-1, 1])

        # Apply the reward shift only during training, not evaluation.
        training_target = terminal_budget - reward_shift

        if method == "MC":
            mc_update(
                theta,
                agent.trajectory,
                training_target,
                alpha
            )

        else:
            agent.finish_episode(training_target)

        if episode % evaluate_every == 0 or episode == n_episodes:
            result = evaluate_policy(
                theta,
                features,
                n_eval_games,
                seed=9000
            )

            checkpoints.append(episode)
            curve.append(result["mean"])

            print(
                f"{method:5} | seed {seed} | "
                f"episode {episode:5}/{n_episodes} "
                f"| greedy validation B10 = {result['mean']:.2f}",
                flush=True
            )

    return (
        theta.copy(),
        np.asarray(checkpoints),
        np.asarray(curve)
    )


def sample_strategy(
    theta: np.ndarray,
    features: str,
    n_games: int,
    seed: int
) -> list[dict]:
    """Collect greedy actions and states for interpretable policy plots."""

    saved_rng = np.random.get_state()
    samples = []

    try:
        np.random.seed(seed)

        for game in range(n_games):
            history = sim_game(
                n=10,
                policy_1=GreedyAgent(theta, features),
                policy_2=default_opponent_policy
            )

            for row in history[:-1]:
                state = row[:7]
                action = row[9:11]

                u, v = action_fractions(
                    state,
                    np.asarray([action])
                )

                samples.append({
                    "game": game,
                    "t": int(row[0]),
                    "B_self": float(row[1]),
                    "B_opp": float(row[2]),
                    "speed_difference": float(row[3] - row[4]),
                    "g": float(row[5]),
                    "R": float(row[6]),
                    "speed_fraction": float(u[0]),
                    "trade_fraction": float(v[0]),
                    "terminal_budget": float(history[-1, 1]),
                })

    finally:
        np.random.set_state(saved_rng)

    return samples


def mean_by_group(
    samples: list[dict],
    key: str,
    indices: np.ndarray,
    n_groups: int
) -> np.ndarray:
    """Group mean, using NaN for bins without observations."""

    values = np.asarray([row[key] for row in samples])
    result = np.full(n_groups, np.nan)

    for group in range(n_groups):
        group_values = values[indices == group]

        if len(group_values):
            result[group] = np.mean(group_values)

    return result


def plot_strategy(
    mc_samples: list[dict],
    td_samples: list[dict],
    output_dir: Path
) -> None:
    """Plot chosen speed/trade fractions against time and signal strength."""

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    signal_edges = np.array([
        0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf
    ])

    signal_labels = [
        "0–0.5", "0.5–1", "1–1.5",
        "1.5–2", "2–3", "3+"
    ]

    for name, samples in [("MC", mc_samples), ("SARSA", td_samples)]:
        t = np.array([row["t"] for row in samples])
        signal = np.array([abs(row["g"]) for row in samples])

        signal_bin = np.searchsorted(
            signal_edges[1:],
            signal,
            side="right"
        )

        for col, metric in enumerate(["speed_fraction", "trade_fraction"]):
            axes[0, col].plot(
                range(10),
                mean_by_group(samples, metric, t, 10),
                marker="o",
                label=name
            )

            axes[1, col].plot(
                range(len(signal_labels)),
                mean_by_group(
                    samples,
                    metric,
                    signal_bin,
                    len(signal_labels)
                ),
                marker="o",
                label=name
            )

    for col, label in enumerate([
        "Speed spending / budget",
        "|Trade| / post-speed wealth"
    ]):
        axes[0, col].set(
            xlabel="Period",
            ylabel=label,
            title=f"{label} by period",
            ylim=(-0.05, 1.05)
        )

        axes[1, col].set(
            xlabel="|Signal| bin",
            ylabel=label,
            title=f"{label} by signal magnitude",
            ylim=(-0.05, 1.05)
        )

        axes[1, col].set_xticks(
            range(len(signal_labels)),
            signal_labels,
            rotation=30
        )

        for row in range(2):
            axes[row, col].grid(alpha=0.25)
            axes[row, col].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "learned_strategy.png", dpi=200)
    plt.close(fig)


def run_experiment(
    features: str = "baseline",
    n_episodes: int = 3000,
    n_seeds: int = 3,
    evaluate_every: int = 300,
    n_eval_games: int = 100,
    n_test_games: int = 400,
    reward_shift: float = 0.0
) -> None:
    """Trains, evaluates and compares MC and SARSA control."""

    if features not in FEATURE_SIZES:
        raise ValueError("features must be 'baseline' or 'quadratic'.")

    reward_name = (
        "wealth"
        if reward_shift == 0
        else f"shift_{reward_shift:g}"
    )

    output_dir = Path("control_results") / f"{features}_{reward_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Starting choices from prediction; tune them for control after first run.
    alpha_settings = {"MC": 0.001, "SARSA": 0.005}

    print(f"Features: {features} ({FEATURE_SIZES[features]} parameters)")
    print(f"Training episodes per method/seed: {n_episodes}; seeds: {n_seeds}")
    print(f"Learning-rate alpha0: {alpha_settings}")
    print("Exploration: max(0.05, 0.30 / (1 + episode/750)^0.8)")

    all_weights = {"MC": [], "SARSA": []}
    all_curves = {"MC": [], "SARSA": []}
    final_rows = []

    # Common evaluation seeds make the methods' comparisons less noisy.
    final_eval_seed = 12000

    default = evaluate_policy(
        None,
        features,
        n_test_games,
        final_eval_seed
    )

    print(
        f"Default vs default: mean B10={default['mean']:.2f}, "
        f"win rate={default['win_rate']:.1%}",
        flush=True
    )

    for method in ("MC", "SARSA"):
        for i in range(n_seeds):
            seed = 123 + i

            theta, checkpoints, curve = train_control(
                method=method,
                features=features,
                n_episodes=n_episodes,
                seed=seed,
                evaluate_every=evaluate_every,
                n_eval_games=n_eval_games,
                alpha_0=alpha_settings[method],
                reward_shift=reward_shift
            )

            all_weights[method].append(theta)
            all_curves[method].append(curve)

            test = evaluate_policy(
                theta,
                features,
                n_test_games,
                final_eval_seed
            )

            final_rows.append({
                "method": method,
                "seed": seed,
                "mean_budget": test["mean"],
                "median_budget": test["median"],
                "budget_se": test["se"],
                "win_rate": test["win_rate"],
                "mean_difference": test["mean_difference"],
            })

            print(
                f"  Held-out {method:5} seed={seed}: "
                f"mean B10={test['mean']:.2f} ± {test['se']:.2f} (SE), "
                f"win rate={test['win_rate']:.1%}",
                flush=True
            )

    # Plot greedy-policy validation performance during learning.
    fig, ax = plt.subplots(figsize=(9, 5))

    for method in ("MC", "SARSA"):
        curves = np.asarray(all_curves[method])

        mean = curves.mean(axis=0)
        sd = curves.std(axis=0)

        line, = ax.plot(
            checkpoints,
            mean,
            label=method
        )

        if n_seeds > 1:
            ax.fill_between(
                checkpoints,
                mean - sd,
                mean + sd,
                color=line.get_color(),
                alpha=0.18
            )

    ax.axhline(
        default["mean"],
        linestyle="--",
        color="grey",
        label="Default vs default (evaluation reference)"
    )

    # Doing nothing always preserves the initial 100; this is a key sanity benchmark.
    ax.axhline(
        100.0,
        linestyle=":",
        color="black",
        label="Hold cash: B10 = 100"
    )

    ax.set(
        xlabel="Training episodes",
        ylabel="Mean greedy-policy terminal budget",
        title=f"Control performance against default ({features})"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "control_learning_curves.png",
        dpi=200
    )
    plt.close(fig)

    # Document the exact exploration and step-size schedules.
    episodes = np.arange(1, n_episodes + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(
        episodes,
        [exploration_rate(int(e)) for e in episodes]
    )

    axes[0].set(
        xlabel="Training episode",
        ylabel="epsilon",
        title="Decreasing exploration rate"
    )

    for method, alpha_0 in alpha_settings.items():
        axes[1].plot(
            episodes,
            [learning_rate(int(e), alpha_0) for e in episodes],
            label=method
        )

    axes[1].set(
        xlabel="Training episode",
        ylabel="alpha",
        title="Decreasing learning rates"
    )

    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "control_schedules.png",
        dpi=200
    )
    plt.close(fig)

    # Inspect the greedy policies from the final training seed.
    mc_samples = sample_strategy(
        all_weights["MC"][-1],
        features,
        100,
        seed=14000
    )

    td_samples = sample_strategy(
        all_weights["SARSA"][-1],
        features,
        100,
        seed=14000
    )

    plot_strategy(mc_samples, td_samples, output_dir)

    with (output_dir / "strategy_samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method"] + list(mc_samples[0])
        )

        writer.writeheader()

        for method, samples in [
            ("MC", mc_samples),
            ("SARSA", td_samples)
        ]:
            for sample in samples:
                writer.writerow({
                    "method": method,
                    **sample
                })

    with (output_dir / "final_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(final_rows[0])
        )

        writer.writeheader()
        writer.writerows(final_rows)

    with (output_dir / "schedules.csv").open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "epsilon",
            "alpha_mc",
            "alpha_sarsa"
        ])

        for episode in episodes:
            writer.writerow([
                int(episode),
                exploration_rate(int(episode)),
                learning_rate(
                    int(episode),
                    alpha_settings["MC"]
                ),
                learning_rate(
                    int(episode),
                    alpha_settings["SARSA"]
                )
            ])

    np.savez(
        output_dir / "trained_weights.npz",
        mc=np.stack(all_weights["MC"]),
        sarsa=np.stack(all_weights["SARSA"]),
        checkpoints=checkpoints,
        mc_validation=np.stack(all_curves["MC"]),
        sarsa_validation=np.stack(all_curves["SARSA"])
    )

    print("\nHeld-out performance (mean over independent training seeds):")

    print(
        f"Default firm 1: B10={default['mean']:.2f} "
        f"± {default['se']:.2f} (SE)"
    )

    print("Hold-cash policy: B10=100.00 (certain, no speed or trade)")

    for method in ("MC", "SARSA"):
        values = np.array([
            row["mean_budget"]
            for row in final_rows
            if row["method"] == method
        ])

        wins = np.array([
            row["win_rate"]
            for row in final_rows
            if row["method"] == method
        ])

        print(
            f"{method:5}: B10={values.mean():.2f} "
            f"± {values.std():.2f} "
            f"(SD across seeds), win rate={wins.mean():.1%}"
        )

    print(
        f"\nSaved figures, CSVs and weights in: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    run_experiment(
        features="baseline",
        n_episodes=3000,
        n_seeds=3,
        evaluate_every=300,
        n_eval_games=100,
        n_test_games=400,
        reward_shift=0.0
    )