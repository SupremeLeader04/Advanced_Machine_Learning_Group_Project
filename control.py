# 2.3.3: On-policy Monte Carlo and SARSA control

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from default_opponent import default_opponent_policy
from environment import sim_game
from features import feasible_actions, state_action_features, state_features

FEATURE_SIZES = {"baseline": 150, "quadratic": 300}

CURVE_VALIDATION_SEED = 9000
SELECTION_VALIDATION_SEED = 10000
FINAL_TEST_SEED = 20000

# learning rate for control (MC and SARSA weight updates)
def learning_rate(
    episode: int, # current training episode
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

# this converts (delta, x) into the 2 spending fractions used by the feature representation
def action_fractions(
    state: list,
    actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return speed-spending and trading-exposure fractions for each action."""

    budget = float(state[1])

    # Each row contains one action: (speed investment, trading position).
    actions = np.asarray(actions, dtype=float).reshape(-1, 2) # converts into a matrix with 2 rows, and the -1 tells it to automatically find row num

    speed_investments = actions[:, 0] # select every row in respective column index 
    trading_positions = actions[:, 1]

    speed_costs = speed_investments ** 2 # total invested into speed so this equals delta squared
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

# Selects which feature representation to use for a particular state and action
def encode_action(
    state: list,
    action: list | tuple,
    features: str
) -> np.ndarray:
    """Features for the chosen action."""

    if features == "baseline":
        return state_action_features(state, action).astype(float) # returns the original 150 element vector

    if features != "quadratic":
        raise ValueError("features must be 'baseline' or 'quadratic'.")

    z = state_features(state).astype(float) # sets z as the 150 element vector

    # encodes u and v as arrays which have a certain speed fraction or trading fraction respectively
    u, v = action_fractions(
        state,
        np.asarray([action], dtype=float)
    )

    u, v = float(u[0]), float(v[0])

    # Retains all original features and adds curvature and an action interaction.
    return np.concatenate([
        z, #50 features vector (one hot)
        u * z, # Scaled by speed fraction
        v * z, #speed by trading fraction
        u * u * z,#scaled by speed fraction squared
        v * v * z,# scaled by trading fraction squared
        u * v * z # scaled by both
    ]) # this returns a 300 dimensional flat array

# predicts the final wealth associated with every supplied action in the current state
def action_values(
    state: list,
    actions: list, 
    theta: np.ndarray, # weights vector
    features: str # whether to use quadratic or linear features
) -> np.ndarray:
    """
    Score all feasible actions without building 111 full feature vectors.

    Q(s,a) = theta.T @ phi(s,a); coefficients depend on the current state.
    This vectorised calculation is identical to scoring each action separately.
    """
    # the weights must match the chosen representation
    if features not in FEATURE_SIZES:
        raise ValueError("Unknown feature representation.")

    if theta.shape != (FEATURE_SIZES[features],):
        raise ValueError("Incorrect parameter vector length.")

    z = state_features(state).astype(float) # one vector with 50 indicators

    coefficients = theta.reshape(-1, 50) @ z  # this reshapes theta so that it containrs 3 blocks of 50 weights
    # this is then multiplied by z to get the 3 dot

    u, v = action_fractions(
        state,
        np.asarray(actions, dtype=float)
    ) # sets ui and v as the repsective speed and trading fractions

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
    # conduct checks to ensure epsilon is feasible
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    # get the list of all of the feasible actions with tuples (delta, x) inside
    actions = feasible_actions(state)


    if epsilon > 0 and np.random.random() < epsilon:
        index = int(np.random.randint(len(actions)))

    else:
        q = action_values(state, actions, theta, features)
        best = np.flatnonzero(q == np.max(q))
        # handling if there are ties
        index = (
            int(np.random.choice(best))
            if random_ties
            else int(best[0])
        )

    action = list(actions[index])

    return action, encode_action(state, action, features)


def mc_update(
    theta: np.ndarray, # current learned weights
    trajectory: list[np.ndarray], # feature cevtors for all state-action pairs visited during an episode
    terminal_budget: float, # firm 1s final budget
    alpha: float # learning rate
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

    prediction = float(theta @ phi) # calculates prediction

    if terminal_budget is not None: # are we in the terminal budget or not
        target = float(terminal_budget)

    else:
        if next_phi is None:
            raise ValueError(
                "A next action feature vector is required before termination."
            )

        target = float(theta @ next_phi) # we need the next state action value

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

# how well does a trained control agent perform against a default opponent over multiple simulated games, simply performance measure
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

    saved_rng = np.random.get_state() #prevents evaluation from consuming random numbers which may be used for later training

    try:
        np.random.seed(seed)

        firm_1 = (
            default_opponent_policy
            if theta is None
            else GreedyAgent(theta, features)
        )

        own, opponent = [], []

        for _ in range(n_games): # for each game, runs for 10 periods, uses firm 1s selected policy, firm2 default
            history = sim_game(
                n=10,
                policy_1=firm_1,
                policy_2=default_opponent_policy
            )

            own.append(float(history[-1, 1]))
            opponent.append(float(history[-1, 2]))

    finally: #executes even if there is an error during evaluation
        np.random.set_state(saved_rng)

    # covert them into arrays
    own = np.asarray(own)
    opponent = np.asarray(opponent)

    return {
        "mean": float(np.mean(own)), #mean terminal wealth
        "median": float(np.median(own)),
        "se": (
            float(np.std(own, ddof=1) / np.sqrt(n_games))
            if n_games > 1
            else 0.0
        ), #standard error
        "win_rate": float(np.mean(own > opponent)), #prop games firm 1 finishes wealthier
        "mean_difference": float(np.mean(own - opponent)), 
        "own": own,
        "opponent": opponent,
    }


def train_control( #the main training function for control
    method: str,
    features: str,
    n_episodes: int,
    seed: int,
    evaluate_every: int,
    n_eval_games: int,
    alpha_0: float,
    epsilon_0: float = 0.30,
    reward_shift: float = 0.0 #changes whether algo learns terminakl wealth or profit relative to 100
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


def mean_by_group( #divide obs into groups and calculate the average value of a chosen variable within each group
    samples: list[dict],
    key: str,
    indices: np.ndarray,
    n_groups: int
) -> np.ndarray:
    """Group mean, using NaN for bins without observations."""

    values = np.asarray([row[key] for row in samples]) #extract the variable we are interested in
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


def _evaluate_candidates(
    candidates: dict,
    n_games: int,
    seed: int,
    output_dir: Path,
    split: str
) -> None:
    """Report per-policy SE and across-training-seed SD as separate quantities."""

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

    print(f"\n{split.capitalize()} evaluation: {n_games} games, seed={seed}")

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

        print(
            f"  {method} seed={training_seed}: mean B10={result['mean']:.2f} "
            f"+/- {result['se']:.2f} (SE over evaluation games)",
            flush=True
        )

    summaries = []

    for method in ["Default", *candidates]:
        selected = [row for row in rows if row["method"] == method]
        means = np.array([row["mean_budget"] for row in selected])
        sd = float(means.std()) if method != "Default" else ""

        summaries.append({
            "method": method,
            "features": selected[0]["features"],
            "alpha_0": selected[0]["alpha_0"],
            "split": split,
            "evaluation_games": n_games,
            "evaluation_seed": seed,
            "training_runs": len(selected) if method != "Default" else 0,
            "mean_budget": float(means.mean()),
            "sd_across_training_seeds": sd
        })

        if method != "Default":
            print(
                f"  {method}: mean B10={means.mean():.2f} +/- {sd:.2f} "
                "(SD across training seeds, not an evaluation SE)"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, data in [
        (f"{split}_evaluation.csv", rows),
        (f"{split}_summary.csv", summaries)
    ]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


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

    print(f"Features: {features} ({FEATURE_SIZES[features]} parameters)")
    print(f"Training episodes per method/seed: {n_episodes}; seeds: {n_seeds}")
    print(f"Learning-rate alpha0: {alpha_settings}")
    print(f"Checkpoint validation: {n_eval_games} games, seed={CURVE_VALIDATION_SEED}")
    print(f"Candidate validation: {n_validation_games} games, seed={SELECTION_VALIDATION_SEED}")
    print("Exploration: max(0.05, 0.30 / (1 + episode/750)^0.8)")

    all_weights = {"MC": [], "SARSA": []}
    all_curves = {"MC": [], "SARSA": []}
    training_seeds = np.arange(123, 123 + n_seeds)

    # Match the benchmark to the games used for the checkpoint curves.
    default_curve = evaluate_policy(
        None,
        features,
        n_eval_games,
        CURVE_VALIDATION_SEED
    )

    print(
        f"Checkpoint default: mean B10={default_curve['mean']:.2f} "
        f"+/- {default_curve['se']:.2f} (SE)",
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

    if make_learning_curve_plot:
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
            default_curve["mean"],
            linestyle="--",
            color="grey",
            label="Default vs default (same checkpoint validation settings)"
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
            ylabel="Mean greedy-policy validation terminal budget",
            title=f"Control validation learning curves ({features})"
        )

        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            output_dir / "control_learning_curves.png",
            dpi=200
        )
        plt.close(fig)

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

    with (output_dir / "schedules.csv").open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "epsilon",
            "alpha_mc",
            "alpha_sarsa"
        ])

        for episode in range(1, n_episodes + 1):
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
    """Compare initial rates on baseline features without using final-test games.

    The consolidated CSV has one row per method/rate/training seed, with
    the candidate's mean and SD repeated alongside individual seed results.
    Its statistics use final candidate validation; plots use checkpoint validation.
    """

    # Figures are saved to disk; no interactive plotting window is needed.
    plt.switch_backend("Agg")

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

        fig, ax = plt.subplots(figsize=(9, 5))

        for alpha_0, curves in zip(learning_rates, learning_curves[method]):
            mean = curves.mean(axis=0)
            sd = curves.std(axis=0)

            line, = ax.plot(checkpoints, mean, label=f"alpha0 = {alpha_0:g}")
            ax.fill_between(
                checkpoints, mean - sd, mean + sd,
                color=line.get_color(), alpha=0.2
            )

        ax.axhline(default_curve_mean, linestyle="--", color="grey",
                   label="Default vs default (checkpoint validation)")
        ax.axhline(100.0, linestyle=":", color="black", label="Hold cash = 100")

        ax.set(
            xlabel="Training episodes",
            ylabel="Mean validation terminal wealth",
            title=f"{method}: learning-rate sensitivity\n"
                  f"{n_eval_games} validation games per checkpoint (seed {CURVE_VALIDATION_SEED})"
        )
        ax.legend()
        ax.grid(alpha=0.25)

        fig.text(
            0.5, 0.02,
            f"Shading: +/- 1 SD across {n_seeds} training seeds, not a confidence interval.",
            ha="center", fontsize=9
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output_dir / f"{method.lower()}_learning_rate_sensitivity.png", dpi=200)
        plt.close(fig)

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
