# 2.3.3: On-policy Monte Carlo and SARSA control

import csv
from pathlib import Path

import numpy as np

from default_opponent import default_opponent_policy
from environment import sim_game
from features import feasible_actions, state_action_features, state_features
from visualisations import (
    plot_control_learning_curves,
    plot_learning_rate_sensitivity,
    plot_strategy,
)

FEATURE_SIZES = {"baseline": 150, "quadratic": 300}

CURVE_VALIDATION_SEED = 9000
SELECTION_VALIDATION_SEED = 10000
FINAL_TEST_SEED = 20000

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
    """Return speed-spending and trading-exposure fractions for each action."""

    budget = float(state[1])

    # Shape: (number of actions, 2), with columns (delta, x).
    actions = np.asarray(actions, dtype=float).reshape(-1, 2)

    speed_investments = actions[:, 0]
    trading_positions = actions[:, 1]

    speed_costs = speed_investments ** 2
    remaining_wealth = np.maximum(0.0, budget - speed_costs)

    # Start at zero so actions with no available budget remain well-defined.
    speed_fractions = np.zeros(len(actions))
    trading_fractions = np.zeros(len(actions))

    if budget > 0:
        speed_fractions = speed_costs / budget

    # Only divide where some wealth remains after speed investment.
    can_trade = remaining_wealth > 0

    trading_fractions[can_trade] = (
        np.abs(trading_positions[can_trade]) / remaining_wealth[can_trade]
    )

    return speed_fractions, trading_fractions

def encode_action(
    state: list,
    action: list | tuple,
    features: str
) -> np.ndarray:
    """Encode an action with 150 baseline or 300 quadratic features."""

    if features == "baseline":
        return state_action_features(state, action).astype(float)

    if features != "quadratic":
        raise ValueError("features must be 'baseline' or 'quadratic'.")

    z = state_features(state).astype(float)  # 50 concatenated state indicators.

    # A single action gives one-element speed and trading fraction arrays.
    u, v = action_fractions(
        state,
        np.asarray([action], dtype=float)
    )

    u, v = float(u[0]), float(v[0])

    # Six blocks of 50 features: retain the baseline terms and add u^2, v^2 and uv.
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
    Score supplied actions without constructing a feature vector for each one.

    Q(s,a) = theta.T @ phi(s,a); coefficients depend on the current state.
    This is algebraically equivalent to scoring each action separately;
    floating-point rounding can differ with the order of operations.
    Values predict terminal wealth minus the reward shift used in training.
    """
    if features not in FEATURE_SIZES:
        raise ValueError("Unknown feature representation.")

    if theta.shape != (FEATURE_SIZES[features],):
        raise ValueError("Incorrect parameter vector length.")

    z = state_features(state).astype(float)

    # Three weight blocks for baseline features, or six for quadratic features.
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
    """Update weights in place using every visited state-action feature vector.

    terminal_budget is the training return, with any reward shift already applied.
    """

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
    """Update weights in place using the next action actually chosen by SARSA.

    Intermediate rewards are zero and gamma is one. At termination,
    terminal_budget supplies the training return after any reward shift.
    """

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
        """Apply the final TD update using the supplied, possibly shifted return."""

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
    Evaluate actual terminal wealth against the default opponent.
    theta=None evaluates the default policy against itself, without learning.

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
    Validation always reports actual wealth; the return is final weights,
    checkpoint episode numbers and mean validation wealth at each checkpoint.
    """

    if method not in ("MC", "SARSA"):
        raise ValueError("method must be 'MC' or 'SARSA'.")

    if features not in FEATURE_SIZES:
        raise ValueError("Unknown feature representation.")

    np.random.seed(seed)

    theta = np.zeros(FEATURE_SIZES[features], dtype=float)

    checkpoints = [0]
    curve = [
        evaluate_policy(theta, features, n_eval_games, seed=CURVE_VALIDATION_SEED)["mean"]
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
                seed=CURVE_VALIDATION_SEED
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
    """Collect greedy-policy states and actions for plots and allocation diagnostics."""

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


def _evaluate_candidates(
    candidates: dict,
    n_games: int,
    seed: int,
    output_dir: Path,
    split: str
) -> None:
    """Save per-policy evaluation SE and across-training-seed SD in one CSV.

    Across-seed SD uses ddof=0 to describe the observed training runs;
    it is distinct from uncertainty across evaluation games.
    """

    rows = []

    # The default policy needs no learned feature representation or weights.
    policies = [("Default", "", "baseline", "", None, "")]

    for method, candidate in candidates.items():
        for training_seed, theta in zip(candidate["training_seeds"], candidate["weights"]):
            policies.append((
                method,
                int(training_seed),
                candidate["features"],
                candidate["alpha_0"],
                theta,
                candidate["weights_file"]
            ))

    for method, training_seed, features, alpha, theta, weights_file in policies:
        result = evaluate_policy(theta, features, n_games, seed)

        rows.append({
            "method": method,
            "training_seed": training_seed,
            "features": features if theta is not None else "",
            "alpha_0": alpha,
            "split": split,
            "evaluation_games": n_games,
            "evaluation_seed": seed,
            "mean_budget": result["mean"],
            "budget_se": result["se"],
            "median_budget": result["median"],
            "win_rate": result["win_rate"],
            "mean_difference": result["mean_difference"],
            "weights_file": weights_file
        })

    # Keep individual results and across-seed statistics in one table.
    for method in ["Default", *candidates]:
        selected = [row for row in rows if row["method"] == method]
        means = np.array([row["mean_budget"] for row in selected])
        for row in selected:
            row.update({
                "training_runs": len(selected) if method != "Default" else 0,
                "mean_across_training_seeds": float(means.mean()),
                "sd_across_training_seeds": float(means.std()) if method != "Default" else ""
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / f"{split}_evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_final_test(
    mc_weights_path: str | Path,
    sarsa_weights_path: str | Path | None = None,
    n_games: int = 2000,
    seed: int = FINAL_TEST_SEED,
    output_dir: str | Path = Path("control_results") / "final_test"
) -> None:
    """Test selected saved configurations only, without training or selection.

    Supply separate archives if MC and SARSA were selected from different
    learning-rate or feature experiments. Each archive contains all training seeds.
    """

    if n_games < 1:
        raise ValueError("n_games must be positive.")

    if seed in (CURVE_VALIDATION_SEED, SELECTION_VALIDATION_SEED):
        raise ValueError("Final testing must use a different seed from validation.")

    if sarsa_weights_path is None:
        sarsa_weights_path = mc_weights_path

    candidates = {}

    for method, key, path in [
        ("MC", "mc", mc_weights_path),
        ("SARSA", "sarsa", sarsa_weights_path)
    ]:
        with np.load(path) as saved:
            required = {key, "features", "training_seeds", f"alpha_{key}"}

            if not required.issubset(saved.files):
                raise ValueError(
                    f"{path} lacks experiment metadata. Use an archive saved "
                    "by the updated run_experiment()."
                )

            features = str(saved["features"].item())
            training_seeds = saved["training_seeds"]
            weights = saved[key]

            if features not in FEATURE_SIZES or weights.shape != (
                len(training_seeds), FEATURE_SIZES[features]
            ):
                raise ValueError(f"Invalid feature representation or weight shape in {path}.")

            if seed in training_seeds:
                raise ValueError("Final testing must use a different seed from training.")

            candidates[method] = {
                "weights": weights,
                "features": features,
                "training_seeds": training_seeds,
                "alpha_0": float(saved[f"alpha_{key}"]),
                "weights_file": str(Path(path).resolve())
            }

    _evaluate_candidates(candidates, n_games, seed, Path(output_dir), split="test")
    print(f"Saved final-test results in {Path(output_dir).resolve()}")


def run_experiment(
    features: str = "baseline",
    n_episodes: int = 3000,
    n_seeds: int = 3,
    evaluate_every: int = 300,
    n_eval_games: int = 200,
    n_validation_games: int = 1000,
    reward_shift: float = 0.0,
    alpha_mc: float = 0.001,
    alpha_sarsa: float = 0.005,
    output_dir: str | Path | None = None,
    make_strategy_plots: bool = False,
    make_learning_curve_plot: bool = True
) -> None:
    """Train candidates and report validation only; never run final testing."""

    if features not in FEATURE_SIZES:
        raise ValueError("features must be 'baseline' or 'quadratic'.")

    if min(n_episodes, n_seeds, evaluate_every, n_eval_games, n_validation_games) < 1:
        raise ValueError("Episode, seed, interval and evaluation counts must be positive.")

    if not all(np.isfinite(alpha) and alpha > 0 for alpha in (alpha_mc, alpha_sarsa)):
        raise ValueError("Initial learning rates must be finite and positive.")

    reward_name = (
        "wealth"
        if reward_shift == 0
        else f"shift_{reward_shift:g}"
    )

    output_dir = Path(output_dir) if output_dir is not None else (
        Path("control_results")
        / f"{features}_{reward_name}_mc{alpha_mc:g}_sarsa{alpha_sarsa:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_settings = {"MC": alpha_mc, "SARSA": alpha_sarsa}

    all_weights = {"MC": [], "SARSA": []}
    all_curves = {"MC": [], "SARSA": []}
    training_seeds = np.arange(123, 123 + n_seeds)

    # Use the same game count and RNG seed as checkpoint validation.
    default_curve = evaluate_policy(
        None,
        features,
        n_eval_games,
        CURVE_VALIDATION_SEED
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

    if make_learning_curve_plot:
        plot_control_learning_curves(
            checkpoints, all_curves, default_curve["mean"], features, output_dir
        )

    if make_strategy_plots:
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

    np.savez(
        output_dir / "trained_weights.npz",
        mc=np.stack(all_weights["MC"]),
        sarsa=np.stack(all_weights["SARSA"]),
        checkpoints=checkpoints,
        mc_validation=np.stack(all_curves["MC"]),
        sarsa_validation=np.stack(all_curves["SARSA"]),
        features=features,
        training_seeds=training_seeds,
        alpha_mc=alpha_mc,
        alpha_sarsa=alpha_sarsa,
        n_episodes=n_episodes,
        reward_shift=reward_shift,
        curve_validation_games=n_eval_games,
        curve_validation_seed=CURVE_VALIDATION_SEED,
        selection_validation_games=n_validation_games,
        selection_validation_seed=SELECTION_VALIDATION_SEED,
        default_curve_mean=default_curve["mean"],
        default_curve_se=default_curve["se"]
    )

    candidates = {
        method: {
            "weights": np.stack(all_weights[method]),
            "features": features,
            "training_seeds": training_seeds,
            "alpha_0": alpha_settings[method],
            "weights_file": str((output_dir / "trained_weights.npz").resolve())
        }
        for method in ("MC", "SARSA")
    }

    _evaluate_candidates(
        candidates, n_validation_games, SELECTION_VALIDATION_SEED,
        output_dir, split="validation"
    )

    print(
        f"\nSaved figures, CSVs and weights in: {output_dir.resolve()}"
    )


def run_learning_rate_sensitivity(
    n_episodes: int = 3000,
    n_seeds: int = 3,
    evaluate_every: int = 300,
    n_eval_games: int = 200,
    n_validation_games: int = 1000,
    output_dir: str | Path = Path("control_results") / "learning_rate_sensitivity"
) -> None:
    """Compare initial rates using baseline features and unshifted rewards.

    The consolidated CSV has one row per method/rate/training seed, with
    the candidate's mean and SD repeated alongside individual seed results.
    Its statistics use final candidate validation; plots use checkpoint validation.
    No independent test games are evaluated.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = [0.001, 0.0025, 0.005]
    consolidated_rows = []
    learning_curves = {"MC": [], "SARSA": []}

    for rate_index, alpha_0 in enumerate(learning_rates):

        candidate_dir = output_dir / f"alpha_{alpha_0:g}"

        # Reuse training, validation, benchmark evaluation and weight exports.
        run_experiment(
            features="baseline",
            n_episodes=n_episodes,
            n_seeds=n_seeds,
            evaluate_every=evaluate_every,
            n_eval_games=n_eval_games,
            n_validation_games=n_validation_games,
            alpha_mc=alpha_0,
            alpha_sarsa=alpha_0,
            reward_shift=0.0,
            output_dir=candidate_dir,
            make_strategy_plots=False,
            make_learning_curve_plot=False
        )

        with (candidate_dir / "validation_evaluation.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        # The same default benchmark is repeated by each experiment; keep it once.
        if rate_index == 0:
            benchmark = next(row for row in rows if row["method"] == "Default")
            consolidated_rows.append({
                **benchmark,
                "mean_across_training_seeds": benchmark["mean_budget"],
                "sd_across_training_seeds": ""
            })

        for method in ("MC", "SARSA"):

            selected = [row for row in rows if row["method"] == method]
            means = np.array([float(row["mean_budget"]) for row in selected])

            for row in selected:
                consolidated_rows.append({
                    **row,
                    "mean_across_training_seeds": float(means.mean()),
                    "sd_across_training_seeds": float(means.std())
                })

        with np.load(candidate_dir / "trained_weights.npz") as saved:
            checkpoints = saved["checkpoints"]
            default_curve_mean = float(saved["default_curve_mean"])
            learning_curves["MC"].append(saved["mc_validation"])
            learning_curves["SARSA"].append(saved["sarsa_validation"])

    with (output_dir / "learning_rate_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(consolidated_rows[0]))
        writer.writeheader()
        writer.writerows(consolidated_rows)

    for method in ("MC", "SARSA"):
        plot_learning_rate_sensitivity(
            method, learning_rates, checkpoints, learning_curves[method],
            output_dir, n_eval_games, CURVE_VALIDATION_SEED, default_curve_mean
        )

    print(f"\nSaved consolidated validation results and sensitivity plots in: {output_dir.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Control validation experiments")
    parser.add_argument(
        "--learning-rate-sensitivity", action="store_true",
        help="Compare initial learning rates for baseline MC and SARSA using validation only."
    )
    args = parser.parse_args()

    if args.learning_rate_sensitivity:
        run_learning_rate_sensitivity()
    else:
        run_experiment()
