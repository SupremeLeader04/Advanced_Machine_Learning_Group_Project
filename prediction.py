# 2.3.2: On-policy Monte Carlo and TD(0) prediction

import csv
from pathlib import Path

import numpy as np

from default_opponent import default_opponent_policy
from environment import sim_game
from features import feasible_actions, state_action_features
from visualisations import plot_prediction_comparison, plot_prediction_learning_rates

OUTPUT_DIR = Path(__file__).resolve().parent / "prediction"


def evaluation_policy(state: list) -> list:
    """ 
    Follows a fixed policy of spending 20% on speed and
    trading 50% of the remaining budget.
    """

    actions = feasible_actions(state)

    if state[1] == 0:
        return list(actions[0])

    speed_fraction_index = 2       # 20% speed spending
    trading_fraction_index = 5     # 50% trading exposure

    # Each speed level has 11 trading choices when the signal is nonzero
    if state[5] == 0:
        action_list_index = speed_fraction_index
    else:
        action_list_index = speed_fraction_index * 11 + trading_fraction_index

    return list(actions[action_list_index])


def episode_data(policy_1=evaluation_policy, policy_2=default_opponent_policy) -> tuple[np.ndarray, float]:
    """Simulates one game and extracts firm 1's 10 state-action feature vectors."""
    history = sim_game(
        n=10,
        policy_1=policy_1,
        policy_2=policy_2
    )

    # The first 10 rows are decision periods; the last row is terminal.
    features = np.vstack([
        state_action_features(row[:7], (row[9], row[10]))
        for row in history[:-1]
    ])

    terminal_budget = float(history[-1, 1])

    return features, terminal_budget


def learning_rate(
    episode: int,
    alpha_0: float,
    decay_scale: float = 500.0,
    exponent: float = 0.6
) -> float:
    """Returns a decreasing step size indexed by the training episode."""
    return alpha_0 / (1.0 + episode / decay_scale) ** exponent


def monte_carlo_update(
    theta: np.ndarray,
    features: np.ndarray,
    terminal_budget: float,
    alpha: float
) -> None:
    """Performs semi-gradient, every-visit MC updates after a complete game."""

    for phi in features:
        prediction = float(theta @ phi)

        theta += alpha * (terminal_budget - prediction) * phi


def td_update(
    theta: np.ndarray,
    features: np.ndarray,
    terminal_budget: float,
    alpha: float
) -> None:
    """Performs on-policy one-step TD(0) action-value prediction updates."""

    # Immediate rewards are 0 and the discount factor is 1.
    for t, phi in enumerate(features):

        prediction = float(theta @ phi)

        if t == len(features) - 1:
            target = terminal_budget
        else:
            target = float(theta @ features[t + 1])

        theta += alpha * (target - prediction) * phi


def validation_data(
    n_games: int = 400,
    seed: int = 2030
) -> tuple[np.ndarray, np.ndarray]:
    """Creates independent held-out episodes for prediction comparison."""

    np.random.seed(seed)

    all_features = []
    all_targets = []

    for _ in range(n_games):

        features, terminal_budget = episode_data()

        all_features.append(features)
        # Every decision in the episode has the same terminal return
        all_targets.extend([terminal_budget] * len(features))

    return np.vstack(all_features), np.array(all_targets, dtype=float)


def prediction_rmse(
    theta: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray
) -> float:
    """Calculates held-out RMSE against realised terminal budgets."""

    errors = features @ theta - targets

    return float(np.sqrt(np.mean(errors**2)))


def train_predictors(
    n_episodes: int,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    seed: int = 123,
    evaluate_every: int = 100,
    alpha_mc: float = 0.001,
    alpha_td: float = 0.005
) -> tuple:
    """Trains MC and TD on the same simulated games using separate learning rates."""

    np.random.seed(seed)

    theta_mc = np.zeros(150, dtype=float)
    theta_td = np.zeros(150, dtype=float)

    checkpoints = [0]

    mc_errors = [
        prediction_rmse(theta_mc, validation_features, validation_targets)
    ]

    td_errors = [
        prediction_rmse(theta_td, validation_features, validation_targets)
    ]

    for episode in range(1, n_episodes + 1):

        # Both methods learn from the same game for a controlled comparison.
        features, terminal_budget = episode_data()

        alpha_mc_n = learning_rate(episode, alpha_0=alpha_mc)
        alpha_td_n = learning_rate(episode, alpha_0=alpha_td)

        monte_carlo_update(
            theta_mc, features, terminal_budget, alpha_mc_n
        )

        td_update(
            theta_td, features, terminal_budget, alpha_td_n
        )

        if episode % evaluate_every == 0 or episode == n_episodes:

            checkpoints.append(episode)

            mc_errors.append(
                prediction_rmse(
                    theta_mc, validation_features, validation_targets
                )
            )

            td_errors.append(
                prediction_rmse(
                    theta_td, validation_features, validation_targets
                )
            )

    return (
        theta_mc,
        theta_td,
        np.array(checkpoints),
        np.array(mc_errors),
        np.array(td_errors)
    )


def run_experiment(
    n_episodes: int = 3000,
    n_validation: int = 400,
    n_seeds: int = 3,
    evaluate_every: int = 100,
) -> None:
    """Compares MC and TD on unseen games and creates figures for the report."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_val, y_val = validation_data(n_validation, seed=2030)

    mc_runs = []
    td_runs = []

    mc_weights = []
    td_weights = []

    for i in range(n_seeds):
        print(f"Training prediction seed {123 + i} ({i + 1}/{n_seeds})", flush=True)

        theta_mc, theta_td, checkpoints, mc_errors, td_errors = (
            train_predictors(
                n_episodes,
                X_val,
                y_val,
                seed=123 + i,
                evaluate_every=evaluate_every
            )
        )

        mc_runs.append(mc_errors)
        td_runs.append(td_errors)
        
        mc_weights.append(theta_mc.copy())
        td_weights.append(theta_td.copy())

    mc_runs = np.array(mc_runs)
    td_runs = np.array(td_runs)

    plot_prediction_comparison(checkpoints, mc_runs, td_runs, OUTPUT_DIR)

    steps = np.arange(1, n_episodes + 1)
    schedules = {
        name: [learning_rate(int(n), alpha_0=alpha_0) for n in steps]
        for name, alpha_0 in [("Monte Carlo", 0.001), ("TD(0)", 0.005)]
    }
    plot_prediction_learning_rates(steps, schedules, OUTPUT_DIR)

    # Save trained weights from every seed for later evaluation.
    np.savez(
        OUTPUT_DIR / "prediction_weights.npz",
        theta_mc=np.array(mc_weights),
        theta_td=np.array(td_weights)
    )

    print(f"Saved prediction figures and weights in: {OUTPUT_DIR}")


def compare_learning_rates(
    n_episodes: int = 3000,
    n_validation: int = 400,
    n_seeds: int = 3,
    output_dir: str | Path | None = None,
) -> None:
    """Compare learning rates on validation games and save two CSV tables."""

    if min(n_episodes, n_validation, n_seeds) < 1:
        raise ValueError("Episodes, validation games and seeds must be positive.")

    output_dir = Path(output_dir) if output_dir is not None else (
        OUTPUT_DIR / "learning_rate_comparison"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(123, 123 + n_seeds))

    rates = {
        "MC": [0.001, 0.0005, 0.00025],
        "TD": [0.005, 0.0075, 0.01]
    }

    X_val, y_val = validation_data(n_games=n_validation, seed=2030)

    seed_rows = []
    summary_rows = []

    # Pairing rates saves simulation work; the learners have separate weights.
    for alpha_mc, alpha_td in zip(rates["MC"], rates["TD"]):

        final_errors = {"MC": [], "TD": []}

        for seed in seeds:

            print(
                f"Training seed {seed}: MC={alpha_mc:g}, TD={alpha_td:g}",
                flush=True
            )

            _, _, _, mc_errors, td_errors = train_predictors(
                n_episodes,
                X_val,
                y_val,
                seed=seed,
                evaluate_every=100,
                alpha_mc=alpha_mc,
                alpha_td=alpha_td
            )

            for method, alpha, errors in [
                ("MC", alpha_mc, mc_errors),
                ("TD", alpha_td, td_errors)
            ]:
                final_rmse = float(errors[-1])
                final_errors[method].append(final_rmse)

                seed_rows.append({
                    "method": method,
                    "alpha_0": alpha,
                    "training_seed": seed,
                    "final_validation_rmse": final_rmse
                })

        for method, alpha in [("MC", alpha_mc), ("TD", alpha_td)]:

            mean = float(np.mean(final_errors[method]))
            sd = float(np.std(final_errors[method]))

            summary_rows.append({
                "method": method,
                "alpha_0": alpha,
                "mean_final_validation_rmse": mean,
                "sd_across_training_seeds": sd
            })

    # Group each table by method, retaining the candidate order above.
    summary_rows.sort(key=lambda row: row["method"])
    seed_rows.sort(key=lambda row: row["method"])

    for filename, rows in [
        ("summary.csv", summary_rows),
        ("per_seed_results.csv", seed_rows)
    ]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:

            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved comparison results in {output_dir.resolve()}")


def evaluate_test_set(
    n_games: int = 2000,
    seed: int = 2031,
    weights_path: str | Path | None = None,
) -> None:
    """Evaluate saved predictors and save a CSV summary of independent-test RMSE.

    By default, load prediction_weights.npz from the prediction folder. Supply
    weights_path if the saved weights are in another folder.
    """
    if n_games < 1:
        raise ValueError("n_games must be positive.")

    weights_path = (
        OUTPUT_DIR / "prediction_weights.npz"
        if weights_path is None
        else Path(weights_path)
    )

    # Load before simulation so a missing file is reported immediately.
    with np.load(weights_path) as weights:
        trained_weights = {
            "Monte Carlo": weights["theta_mc"],
            "TD(0)": weights["theta_td"],
        }

    # The data generator uses the same fixed policy with a fresh seed.
    X_test, y_test = validation_data(n_games=n_games, seed=seed)

    baseline_rmse = float(np.sqrt(np.mean((100.0 - y_test) ** 2)))

    summary_rows = []

    for name, weights in trained_weights.items():
        errors = np.array([
            prediction_rmse(theta, X_test, y_test)
            for theta in weights
        ])

        summary_rows.append({
            "method": name,
            "mean_test_rmse": float(errors.mean()),
            "sd_across_training_seeds": float(errors.std()),
            "training_runs": len(errors),
            "test_games": n_games,
            "test_seed": seed
        })

    # The constant predictor has no training runs or across-seed variation.
    summary_rows.append({
        "method": "Constant-100",
        "mean_test_rmse": baseline_rmse,
        "sd_across_training_seeds": "",
        "training_runs": 0,
        "test_games": n_games,
        "test_seed": seed
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "prediction_test_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved test summary: {summary_path.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prediction experiments")
    parser.add_argument("--compare-rates", action="store_true",
                        help="Run the targeted learning-rate comparison and export CSVs.")
    args = parser.parse_args()
    if args.compare_rates:
        compare_learning_rates()
    else:
        run_experiment()
        evaluate_test_set()

