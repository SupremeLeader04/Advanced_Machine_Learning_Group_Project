# 2.3.2: On-policy Monte Carlo and TD(0) prediction

import numpy as np
import matplotlib.pyplot as plt

from environment import sim_game
from default_opponent import default_opponent_policy
from features import feasible_actions, state_action_features


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


def safe_default_opponent(state: list) -> list:
    """Runs the default opponent, correcting machine-precision overspending."""
    delta, x = map(float, default_opponent_policy(state))
    B = float(state[1])

    # The environment rejects tiny floating-point overspending.
    while delta**2 > B:
        delta = float(np.nextafter(delta, 0.0))

    while delta**2 + abs(x) > B:
        x = float(np.nextafter(x, 0.0))

    return [delta, x]


def episode_data() -> tuple[np.ndarray, float]:
    """Simulates one game and extracts firm 1's 10 state-action feature vectors."""
    history = sim_game(
        n=10,
        policy_1=evaluation_policy,
        policy_2=safe_default_opponent
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

    X_val, y_val = validation_data(n_validation, seed=2030)

    mc_runs = []
    td_runs = []

    mc_weights = []
    td_weights = []

    for i in range(n_seeds):

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

    print("Fixed policy: 20% speed spending, 50% trading exposure")
    print(f"Training: {n_episodes} games per seed; {n_seeds} independent seeds")
    print(f"Validation: {n_validation} independent games")

    baseline_rmse = np.sqrt(np.mean((100 - y_val)**2))

    print(f"Constant-100 RMSE: {baseline_rmse:.3f}")
    print(
        f"MC final RMSE: {mc_runs[:, -1].mean():.3f} "
        f"± {mc_runs[:, -1].std():.3f}"
    )
    print(
        f"TD final RMSE: {td_runs[:, -1].mean():.3f} "
        f"± {td_runs[:, -1].std():.3f}"
    )

    print("\nStep-size schedule: alpha(n) = alpha_0 / (1 + n/500)^0.6")

    for name, alpha_0 in [
        ("Monte Carlo", 0.001),
        ("TD(0)", 0.005)
    ]:
        print(f"\n{name} (alpha_0 = {alpha_0}):")

        for n in [1, 100, 500, 1000, n_episodes]:
            print(
                f"Episode {n:5d}: "
                f"alpha = {learning_rate(n, alpha_0=alpha_0):.6f}"
            )

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, values in [
        ("Monte Carlo", mc_runs),
        ("TD(0)", td_runs)
    ]:

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        line, = ax.plot(checkpoints, mean, label=name)

        ax.fill_between(
            checkpoints,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.2
        )

    ax.set(
        xlabel="Training episodes",
        ylabel="Held-out RMSE (terminal budget units)",
        title="On-policy prediction: Monte Carlo vs TD(0)"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("prediction_comparison.png", dpi=200)
    plt.close(fig)

    steps = np.arange(1, n_episodes + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the separate learning-rate schedules used for MC and TD.
    for name, alpha_0 in [
        ("Monte Carlo", 0.001),
        ("TD(0)", 0.005)
    ]:
        ax.plot(
            steps,
            [learning_rate(int(n), alpha_0=alpha_0) for n in steps],
            label=name
        )

    ax.set(
        xlabel="Training episode",
        ylabel="Step size alpha",
        title="Decreasing learning-rate schedules"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("prediction_learning_rate.png", dpi=200)
    plt.close(fig)
    
    # Save trained weights from every seed for later evaluation.
    np.savez(
        "prediction_weights.npz",
        theta_mc=np.array(mc_weights),
        theta_td=np.array(td_weights)
    )

    print(
        "\nSaved: prediction_comparison.png, "
        "prediction_learning_rate.png, prediction_weights.npz"
    )


if __name__ == "__main__":
    run_experiment()

