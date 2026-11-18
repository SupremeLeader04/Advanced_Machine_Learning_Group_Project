"""On-policy Monte Carlo and SARSA control experiments."""

from pathlib import Path

import numpy as np

from default_opponent import default_opponent_policy
from environment import sim_game
from features import feasible_actions, state_action_features, state_features
from prediction import learning_rate
from visualisations import (
    plot_control_final_comparison,
    plot_strategy,
    save_csv,
)

CURVE_VALIDATION_SEED = 9000
SELECTION_VALIDATION_SEED = 10000
FINAL_TEST_SEED = 20000


def exploration_rate(
    episode: int,
    epsilon_0: float = 0.30,
    decay_scale: float = 750.0,
    exponent: float = 0.8,
) -> float:
    """Decay epsilon-greedy exploration toward zero."""
    if episode < 0:
        raise ValueError("episode must be non-negative.")
    return epsilon_0 / (1.0 + episode / decay_scale) ** exponent


def epsilon_greedy(
    state: list,
    theta: np.ndarray,
    epsilon: float,
    random_ties: bool = True,
) -> list:
    """Choose a feasible action using the linear state-action value function."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    actions = feasible_actions(state)
    if epsilon > 0.0 and np.random.random() < epsilon:
        return list(actions[np.random.randint(len(actions))])

    if theta.shape != (150,):
        raise ValueError("theta must contain 150 baseline feature weights.")

    z = state_features(state).astype(float)
    action_array = np.asarray(actions, dtype=float).reshape(-1, 2)
    budget = float(state[1])
    speed_cost = action_array[:, 0] ** 2
    speed_fraction = speed_cost / budget if budget > 0 else np.zeros(len(actions))
    remaining = np.maximum(0.0, budget - speed_cost)
    trade_fraction = np.divide(
        np.abs(action_array[:, 1]),
        remaining,
        out=np.zeros(len(actions)),
        where=remaining > 0,
    )

    coefficients = theta.reshape(3, 50) @ z
    values = (
        coefficients[0]
        + coefficients[1] * speed_fraction
        + coefficients[2] * trade_fraction
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Nonfinite action values.")

    best = np.flatnonzero(values == np.max(values))
    index = np.random.choice(best) if random_ties else best[0]
    return list(actions[index])


def mc_update(
    theta: np.ndarray,
    trajectory: list[np.ndarray],
    terminal_budget: float,
    alpha: float,
) -> None:
    """Apply every-visit semi-gradient Monte Carlo updates in place."""
    for phi in trajectory:
        theta += alpha * (terminal_budget - float(theta @ phi)) * phi


def sarsa_update(
    theta: np.ndarray,
    phi: np.ndarray,
    next_phi: np.ndarray | None,
    alpha: float,
    terminal_budget: float | None = None,
) -> None:
    """Apply one on-policy SARSA update; intermediate rewards are zero."""
    prediction = float(theta @ phi)
    if terminal_budget is not None:
        target = float(terminal_budget)
    elif next_phi is not None:
        target = float(theta @ next_phi)
    else:
        raise ValueError("A next action feature vector is required before termination.")
    theta += alpha * (target - prediction) * phi


def _greedy_policy(theta: np.ndarray):
    def policy(state: list) -> list:
        return epsilon_greedy(state, theta, epsilon=0.0, random_ties=False)
    return policy


def _print_summary_table(title: str, rows: list[dict], columns: list[str]) -> None:
    """Print a compact table to stdout for stage-wise reporting."""
    if not rows:
        return
    print(f"\n{title}")
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in columns}
    header = " | ".join(key.ljust(widths[key]) for key in columns)
    sep = "-+-".join("-" * widths[key] for key in columns)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(key, "")).ljust(widths[key]) for key in columns))


def evaluate_policy(
    theta: np.ndarray | None,
    features: str,
    n_games: int,
    seed: int,
) -> dict:
    """Evaluate terminal wealth against the default opponent without altering RNG state."""
    if features != "baseline":
        raise ValueError("Only the baseline 150-feature representation is supported.")
    if n_games < 1:
        raise ValueError("n_games must be positive.")

    saved_rng = np.random.get_state()
    try:
        np.random.seed(seed)
        policy = default_opponent_policy if theta is None else _greedy_policy(theta)
        own, opponent = [], []
        for _ in range(n_games):
            history = sim_game(
                n=10, policy_1=policy, policy_2=default_opponent_policy
            )
            own.append(float(history[-1, 1]))
            opponent.append(float(history[-1, 2]))
    finally:
        np.random.set_state(saved_rng)

    own = np.asarray(own)
    opponent = np.asarray(opponent)
    return {
        "mean": float(own.mean()),
        "median": float(np.median(own)),
        "se": float(own.std(ddof=1) / np.sqrt(n_games)) if n_games > 1 else 0.0,
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
    reward_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train baseline-feature MC or SARSA control against the default opponent."""
    if method not in ("MC", "SARSA"):
        raise ValueError("method must be 'MC' or 'SARSA'.")
    if features != "baseline":
        raise ValueError("Only the baseline 150-feature representation is supported.")
    if min(n_episodes, evaluate_every, n_eval_games) < 1:
        raise ValueError("Episode, evaluation interval and game counts must be positive.")
    if not np.isfinite(alpha_0) or alpha_0 <= 0:
        raise ValueError("alpha_0 must be finite and positive.")
    if not 0.0 < epsilon_0 <= 1.0:
        raise ValueError("epsilon_0 must be in (0, 1].")

    np.random.seed(seed)
    theta = np.zeros(150, dtype=float)
    checkpoints = [0]
    curve = [evaluate_policy(
        theta, features, n_eval_games, CURVE_VALIDATION_SEED
    )["mean"]]

    for episode in range(1, n_episodes + 1):
        alpha = learning_rate(episode, alpha_0=alpha_0)
        epsilon = exploration_rate(episode, epsilon_0=epsilon_0)
        trajectory = []
        agent_state = {"previous_phi": None}

        def policy(state: list) -> list:
            action = epsilon_greedy(state, theta, epsilon)
            phi = state_action_features(state, action).astype(float)
            if method == "MC":
                trajectory.append(phi)
            elif agent_state["previous_phi"] is not None:
                sarsa_update(theta, agent_state["previous_phi"], phi, alpha)
            agent_state["previous_phi"] = phi
            return action

        history = sim_game(
            n=10, policy_1=policy, policy_2=default_opponent_policy
        )
        training_return = float(history[-1, 1]) - reward_shift
        if method == "MC":
            mc_update(theta, trajectory, training_return, alpha)
        else:
            sarsa_update(
                theta,
                agent_state["previous_phi"],
                None,
                alpha,
                terminal_budget=training_return,
            )

        if episode % evaluate_every == 0 or episode == n_episodes:
            result = evaluate_policy(
                theta, features, n_eval_games, CURVE_VALIDATION_SEED
            )
            checkpoints.append(episode)
            curve.append(result["mean"])
            print(
                f"{method:5} | seed {seed} | episode {episode}/{n_episodes} "
                f"| greedy validation wealth = {result['mean']:.2f}",
                flush=True,
            )

    return theta.copy(), np.asarray(checkpoints), np.asarray(curve)


def sample_strategy(
    theta: np.ndarray,
    n_games: int = 100,
    seed: int = 14000,
) -> list[dict]:
    """Sample a greedy policy for visualizing its budget allocations."""
    saved_rng = np.random.get_state()
    samples = []
    try:
        np.random.seed(seed)
        policy = _greedy_policy(theta)
        for game in range(n_games):
            history = sim_game(
                n=10, policy_1=policy, policy_2=default_opponent_policy
            )
            for row in history[:-1]:
                budget, delta, trade = float(row[1]), float(row[9]), float(row[10])
                remaining = max(0.0, budget - delta**2)
                samples.append({
                    "game": game,
                    "t": int(row[0]),
                    "g": float(row[5]),
                    "speed_fraction": delta**2 / budget if budget > 0 else 0.0,
                    "trade_fraction": abs(trade) / remaining if remaining > 0 else 0.0,
                })
    finally:
        np.random.set_state(saved_rng)
    return samples


def run_experiment(
    n_episodes: int = 3000,
    n_seeds: int = 3,
    evaluate_every: int = 300,
    n_eval_games: int = 100,
    n_validation_games: int = 300,
    n_test_games: int = 1000,
    n_screen_episodes: int = 750,
    n_screen_games: int = 100,
    n_refine: int = 2,
    alpha_values: dict | None = None,
    epsilon_values: list[float] | None = None,
    output_dir: str | Path = Path("control_results") / "on_policy_control",
) -> None:
    """Screen hyperparameters, refine the best candidates, and compare policies."""
    if min(n_episodes, n_seeds, evaluate_every, n_eval_games,
           n_validation_games, n_test_games, n_screen_episodes,
           n_screen_games, n_refine) < 1:
        raise ValueError("Episode, seed, interval and evaluation counts must be positive.")

    alpha_values = alpha_values or {
        "MC": [0.001, 0.0005, 0.00025],
        "SARSA": [0.005, 0.0075, 0.01],
    }
    epsilon_values = epsilon_values or [0.1, 0.3, 0.5]
    if any(method not in alpha_values for method in ("MC", "SARSA")):
        raise ValueError("alpha_values must contain MC and SARSA candidate lists.")
    if any(
        not alpha_values[method]
        or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alpha_values[method])
        for method in ("MC", "SARSA")
    ):
        raise ValueError("Each method needs finite, positive alpha candidates.")
    if not epsilon_values:
        raise ValueError("epsilon_values must contain at least one candidate.")
    if any(not 0.0 < value <= 1.0 for value in epsilon_values):
        raise ValueError("epsilon_values must contain values in (0, 1].")
    n_refine = min(
        n_refine,
        min(len(alpha_values[method]) * len(epsilon_values)
            for method in ("MC", "SARSA")),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_seeds = list(range(123, 123 + n_seeds))
    screening_scores = {
        method: np.full(
            (len(alpha_values[method]), len(epsilon_values)),
            np.nan,
            dtype=float,
        )
        for method in ("MC", "SARSA")
    }
    refinement_scores = {
        method: np.full_like(screening_scores[method], np.nan)
        for method in ("MC", "SARSA")
    }
    screening_rows = []
    refinement_rows = []
    rows_by_candidate = {}
    results = {}

    for method in ("MC", "SARSA"):
        for i, alpha_0 in enumerate(alpha_values[method]):
            for j, epsilon_0 in enumerate(epsilon_values):
                print(
                    f"Screening {method}: alpha_0={alpha_0:g}, "
                    f"epsilon_0={epsilon_0:g} ({i * len(epsilon_values) + j + 1}/"
                    f"{len(alpha_values[method]) * len(epsilon_values)})",
                    flush=True,
                )
                _, _, curve = train_control(
                    method,
                    "baseline",
                    n_screen_episodes,
                    123,
                    n_screen_episodes,
                    n_screen_games,
                    alpha_0,
                    epsilon_0,
                )
                mean = float(curve[-1])
                screening_scores[method][i, j] = mean
                row = {
                    "method": method,
                    "alpha_0": alpha_0,
                    "epsilon_0": epsilon_0,
                    "screening_mean_validation_wealth": mean,
                    "screening_seed": 123,
                    "screening_episodes": n_screen_episodes,
                    "refinement_mean_validation_wealth": "",
                    "refinement_sd_across_seeds": "",
                    "refinement_validation_games": "",
                    "refinement_seeds": "",
                    "selected": False,
                }
                screening_rows.append({
                    "method": method,
                    "alpha_0": alpha_0,
                    "epsilon_0": epsilon_0,
                    "screening_mean_validation_wealth": mean,
                    "screening_seed": 123,
                    "screening_episodes": n_screen_episodes,
                })
                rows_by_candidate[(method, i, j)] = row

        ranked = np.argsort(screening_scores[method], axis=None)[::-1]
        finalists = [
            np.unravel_index(index, screening_scores[method].shape)
            for index in ranked[:n_refine]
        ]
        print(
            f"Screening shortlisted {method}: "
            + ", ".join(
                f"alpha_0={alpha_values[method][i]:g}, "
                f"epsilon_0={epsilon_values[j]:g}"
                for i, j in finalists
            ),
            flush=True,
        )

        best_mean = -np.inf
        method_result = None
        best_candidate = None
        for i, j in finalists:
            alpha_0, epsilon_0 = alpha_values[method][i], epsilon_values[j]
            candidate_weights, candidate_curves = [], []
            candidate_validation_scores = []
            candidate_checkpoints = None
            for seed in training_seeds:
                print(
                    f"Refining {method}: alpha_0={alpha_0:g}, "
                    f"epsilon_0={epsilon_0:g}, seed={seed}",
                    flush=True,
                )
                theta, candidate_checkpoints, curve = train_control(
                    method,
                    "baseline",
                    n_episodes,
                    seed,
                    evaluate_every,
                    n_eval_games,
                    alpha_0,
                    epsilon_0,
                )
                candidate_weights.append(theta)
                candidate_curves.append(curve)
                validation = evaluate_policy(
                    theta,
                    "baseline",
                    n_validation_games,
                    SELECTION_VALIDATION_SEED,
                )
                candidate_validation_scores.append(validation["mean"])

            candidate_curves_array = np.stack(candidate_curves)
            validation_scores = np.asarray(candidate_validation_scores)
            mean = float(validation_scores.mean())
            row = rows_by_candidate[(method, i, j)]
            row["refinement_mean_validation_wealth"] = mean
            row["refinement_sd_across_seeds"] = float(validation_scores.std())
            row["refinement_validation_games"] = n_validation_games
            row["refinement_seeds"] = n_seeds
            refinement_scores[method][i, j] = mean
            refinement_rows.append({
                "method": method,
                "alpha_0": alpha_0,
                "epsilon_0": epsilon_0,
                "refinement_mean_validation_wealth": mean,
                "refinement_sd_across_seeds": float(validation_scores.std()),
                "refinement_validation_games": n_validation_games,
                "refinement_seeds": n_seeds,
            })
            if mean > best_mean:
                best_mean = mean
                best_candidate = (i, j)
                method_result = {
                    "alpha_0": alpha_0,
                    "epsilon_0": epsilon_0,
                    "weights": candidate_weights,
                    "curves": candidate_curves_array,
                    "validation_scores": validation_scores,
                    "checkpoints": candidate_checkpoints,
                }

        if method_result is None or best_candidate is None:
            raise RuntimeError(f"No refined candidate was produced for {method}.")
        best_row = rows_by_candidate[(method, *best_candidate)]
        best_row["selected"] = True
        print(
            f"Selected {method}: alpha_0={method_result['alpha_0']:g}, "
            f"epsilon_0={method_result['epsilon_0']:g}, "
            f"validation wealth={best_mean:.2f}",
            flush=True,
        )
        # Keep refined candidate runs as the final training runs; do not retrain them.
        results[method] = method_result

    save_csv(output_dir / "screening_results.csv", screening_rows)
    save_csv(output_dir / "refinement_results.csv", refinement_rows)

    _print_summary_table(
        "Screening results (validation wealth after 750 episodes)",
        screening_rows,
        ["method", "alpha_0", "epsilon_0", "screening_mean_validation_wealth", "screening_seed", "screening_episodes"],
    )
    _print_summary_table(
        "Refinement results (mean validation wealth across seeds)",
        refinement_rows,
        ["method", "alpha_0", "epsilon_0", "refinement_mean_validation_wealth", "refinement_sd_across_seeds", "refinement_validation_games", "refinement_seeds"],
    )

    selected = {
        method: (results[method]["alpha_0"], results[method]["epsilon_0"])
        for method in ("MC", "SARSA")
    }
    weights = {
        method: results[method]["weights"]
        for method in ("MC", "SARSA")
    }

    final_indices = {
        method: int(np.argmax(results[method]["validation_scores"]))
        for method in ("MC", "SARSA")
    }
    policy_samples = {
        method: sample_strategy(
            weights[method][final_indices[method]], seed=14000
        )
        for method in ("MC", "SARSA")
    }
    plot_strategy(policy_samples["MC"], policy_samples["SARSA"], output_dir)

    evaluation_rows = []
    test_wealth = {}
    for method in ("MC", "SARSA"):
        alpha_0, epsilon_0 = selected[method]
        result = evaluate_policy(
            weights[method][final_indices[method]],
            "baseline",
            n_test_games,
            FINAL_TEST_SEED,
        )
        test_wealth[method] = result["own"]
        evaluation_rows.append({
            "method": method,
            "alpha_0": alpha_0,
            "epsilon_0": epsilon_0,
            "training_seed": training_seeds[final_indices[method]],
            "test_games": n_test_games,
            "test_seed": FINAL_TEST_SEED,
            "mean_wealth": result["mean"],
            "wealth_se": result["se"],
            "median_wealth": result["median"],
            "win_rate_vs_default": result["win_rate"],
        })

    save_csv(output_dir / "final_test_results.csv", evaluation_rows)
    _print_summary_table(
        "Final held-out test results (same unseen test set for both methods)",
        evaluation_rows,
        ["method", "alpha_0", "epsilon_0", "mean_wealth", "wealth_se", "median_wealth", "win_rate_vs_default"],
    )
    plot_control_final_comparison(test_wealth, output_dir)
    print(f"Saved control results in {output_dir.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-policy MC and SARSA control")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--screen-episodes", type=int, default=750)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_experiment(
        n_episodes=args.episodes,
        n_seeds=args.seeds,
        n_screen_episodes=args.screen_episodes,
        output_dir=args.output_dir or Path("control_results") / "on_policy_control",
    )
