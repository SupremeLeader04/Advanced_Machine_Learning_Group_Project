"""Plot precomputed experiment results; never train, evaluate or sample policies.

Shared home for Prediction and Control experiment visualisations.

Regenerate an experiment's learning curves without retraining::

    regenerate_control_figures("path/to/trained_weights.npz")

Pass make_strategy_plots=True to also read strategy_samples.csv from the
archive's folder. If samples were not saved, they cannot be reconstructed
from weights alone without policy sampling; this module never samples.

Regenerate one method's learning-rate comparison::

    regenerate_learning_rate_sensitivity(
        ["rate1/trained_weights.npz", "rate2/trained_weights.npz"],
        method="SARSA", output_dir="comparison"
    )

An optional output_dir for regeneration keeps the original figures untouched.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def save_csv(path: str | Path, rows: list[dict]) -> None:
    """Write experiment rows to CSV, using the first row's keys as columns."""
    if not rows:
        raise ValueError("Cannot save an empty CSV result.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean_interval(
    values: np.ndarray,
    seed: int,
    n_resamples: int = 5000,
) -> tuple[float, float]:
    """Return a reproducible percentile-bootstrap interval for a sample mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite vector with at least two observations.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    batch_size = 250
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)

    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def plot_strategy(
    mc_samples: list[dict],
    td_samples: list[dict],
    output_dir: Path
) -> None:
    """Plot chosen speed/trade fractions against time and signal strength."""

    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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


def plot_control_tuning_heatmaps(
    alpha_values: dict,
    epsilon_values: list[float],
    scores: dict,
    output_dir: str | Path,
    stage: str = "screening",
) -> None:
    """Plot validation wealth for screening or shortlisted refinement runs."""
    if stage not in ("screening", "refinement"):
        raise ValueError("stage must be 'screening' or 'refinement'.")
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    score_arrays = [
        np.asarray(scores[method], dtype=float)
        for method in ("MC", "SARSA")
    ]
    finite_scores = np.concatenate([
        values[np.isfinite(values)] for values in score_arrays
    ])
    if finite_scores.size == 0:
        raise ValueError("At least one finite tuning score is required.")
    vmin = float(finite_scores.min())
    vmax = float(finite_scores.max())
    if vmin == vmax:
        vmax = vmin + 1.0

    for ax, method in zip(axes, ("MC", "SARSA")):
        values = np.asarray(scores[method])
        image = ax.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(
            title=f"{method} {stage} validation wealth",
            xlabel="Initial exploration rate (epsilon_0)",
            ylabel="Initial step size (alpha_0)",
            xticks=np.arange(len(epsilon_values)),
            xticklabels=[f"{value:g}" for value in epsilon_values],
            yticks=np.arange(len(alpha_values[method])),
            yticklabels=[f"{value:g}" for value in alpha_values[method]],
        )
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not np.isfinite(values[i, j]):
                    continue
                label = (
                    f"{values[i, j]:.1f}"
                    if abs(values[i, j]) >= 1
                    else f"{values[i, j]:.2g}"
                )
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color=(
                        "black"
                        if np.dot(
                            plt.get_cmap("viridis")(
                                (values[i, j] - vmin) / (vmax - vmin)
                            )[:3],
                            [0.2126, 0.7152, 0.0722],
                        ) > 0.55
                        else "white"
                    ),
                )
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.16, top=0.9, wspace=0.3)
    colorbar_ax = fig.add_axes([0.91, 0.2, 0.025, 0.65])
    fig.colorbar(image, cax=colorbar_ax, label="Mean terminal wealth")
    filename = (
        "control_hyperparameter_heatmaps.png"
        if stage == "screening"
        else "control_hyperparameter_refinement.png"
    )
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_control_schedules(
    episodes: np.ndarray,
    learning_rates: dict[str, list[float]],
    exploration_rates: dict[str, list[float]],
    output_dir: str | Path,
) -> None:
    """Plot the selected learning and exploration schedules for each method."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for method in ("MC", "SARSA"):
        axes[0].plot(episodes, learning_rates[method], label=method)
        axes[1].plot(episodes, exploration_rates[method], label=method)

    axes[0].set(
        xlabel="Training episode",
        ylabel="Step size alpha",
        title="Selected learning-rate schedules",
    )
    axes[1].set(
        xlabel="Training episode",
        ylabel="Exploration rate epsilon",
        title="Selected exploration schedules",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "control_schedules.png", dpi=200)
    plt.close(fig)


def plot_control_final_comparison(
    test_wealth: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    """Compare test-game mean wealth and show uncertainty separately from dispersion."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    methods = ("MC", "SARSA")
    colors = ["#1f77b4", "#ff7f0e"]
    interval_upper = 100.0
    for position, (method, color) in enumerate(zip(methods, colors), start=1):
        values = np.asarray(test_wealth[method], dtype=float)
        if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{method} test wealth must be a finite vector with at least two games."
            )

        mean = float(values.mean())
        median = float(np.median(values))
        sd = float(values.std(ddof=1))
        lower, upper = _bootstrap_mean_interval(values, seed=2026 + position)
        ax.vlines(position, lower, upper, color=color, linewidth=2)
        ax.hlines([lower, upper], position - 0.035, position + 0.035, color=color)
        ax.scatter(
            position,
            mean,
            color=color,
            marker="o",
            s=55,
            zorder=3,
            label="Mean (95% bootstrap interval)" if position == 1 else None,
        )
        ax.scatter(
            position,
            median,
            color=color,
            marker="D",
            s=40,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label="Median" if position == 1 else None,
        )
        interval_upper = max(interval_upper, upper)
        ax.text(
            position,
            0.97,
            (
                f"Mean {mean:.1f}  |  Median {median:.1f}\n"
                f"SD {sd:.1f}  |  Max {values.max():.1f}\n"
                f"Games {len(values)}"
            ),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.axhline(
        100.0,
        linestyle=":",
        color="black",
        linewidth=1.2,
        label="Hold cash: terminal wealth = 100",
    )
    y_max = max(125.0, interval_upper * 1.12)
    ax.set(
        xticks=np.arange(1, len(methods) + 1),
        xticklabels=methods,
        xlim=(0.5, len(methods) + 0.5),
        xlabel="Control method",
        ylabel="Mean terminal wealth",
        title="Held-out control performance",
        ylim=(80.0, y_max),
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.5,
        0.015,
        "Whiskers: 95% percentile-bootstrap interval for the mean. SD describes test-game spread.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "control_final_comparison.png", dpi=200)
    plt.close(fig)


def regenerate_control_figures(
    weights_path: str | Path,
    output_dir: str | Path | None = None,
    make_learning_curve_plot: bool = True,
    make_strategy_plots: bool = False
) -> None:
    """Redraw saved curves and, optionally, strategy_samples.csv; never simulate."""

    weights_path = Path(weights_path)
    output_dir = Path(output_dir) if output_dir is not None else weights_path.parent

    if make_learning_curve_plot:
        with np.load(weights_path) as saved:
            curves = {
                method: saved[f"{method.lower()}_validation"]
                for method in ("MC", "SARSA")
                if f"{method.lower()}_validation" in saved
            }
            if not curves:
                raise ValueError("The archive contains no saved validation curves.")

            benchmark = (
                float(saved["default_curve_mean"])
                if "default_curve_mean" in saved else None
            )
            plot_control_learning_curves(
                saved["checkpoints"], curves, benchmark,
                str(saved["features"]), output_dir
            )

    if make_strategy_plots:
        # Saved weights alone do not contain the states/actions needed here.
        with (weights_path.parent / "strategy_samples.csv").open(newline="") as file:
            samples = {"MC": [], "SARSA": []}
            for row in csv.DictReader(file):
                method = row.pop("method")
                sample = {key: float(value) for key, value in row.items()}
                samples[method].append(sample)

        plot_strategy(samples["MC"], samples["SARSA"], output_dir)


def regenerate_learning_rate_sensitivity(
    weights_paths: list[str | Path],
    method: str,
    output_dir: str | Path
) -> None:
    """Redraw one sensitivity figure from compatible saved experiment archives."""

    if method not in ("MC", "SARSA"):
        raise ValueError("method must be MC or SARSA.")
    if not weights_paths:
        raise ValueError("Provide at least one saved experiment archive.")

    learning_rates, curves = [], []
    reference = None
    benchmark = None

    for path in weights_paths:
        with np.load(path) as saved:
            settings = (
                str(saved["features"]), float(saved["reward_shift"]),
                int(saved["curve_validation_games"]),
                int(saved["curve_validation_seed"]),
                tuple(saved["training_seeds"]), tuple(saved["checkpoints"])
            )
            if reference is not None and settings != reference:
                raise ValueError("Archives must share features, reward shift and validation design.")
            reference = settings
            learning_rates.append(float(saved[f"alpha_{method.lower()}"]))
            curves.append(saved[f"{method.lower()}_validation"])
            if "default_curve_mean" in saved:
                benchmark = float(saved["default_curve_mean"])

    order = np.argsort(learning_rates)
    plot_learning_rate_sensitivity(
        method, [learning_rates[i] for i in order], np.asarray(reference[5]),
        [curves[i] for i in order], Path(output_dir),
        reference[2], reference[3], benchmark
    )


def plot_control_learning_curves(
    checkpoints: np.ndarray,
    learning_curves: dict[str, np.ndarray],
    default_curve_mean: float | None,
    features: str,
    output_dir: Path
) -> None:
    """Plot per-seed checkpoint wealth arrays, with shape (seeds, checkpoints)."""

    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Plot greedy-policy validation performance during learning.
    fig, ax = plt.subplots(figsize=(9, 5))

    for method in learning_curves:
        curves = np.asarray(learning_curves[method])

        mean = curves.mean(axis=0)
        sd = curves.std(axis=0)

        line, = ax.plot(
            checkpoints,
            mean,
            label=method
        )

        if len(curves) > 1:
            ax.fill_between(
                checkpoints,
                mean - sd,
                mean + sd,
                color=line.get_color(),
                alpha=0.18
            )

    if default_curve_mean is not None:
        ax.axhline(
            default_curve_mean,
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

    fig.text(
        0.5,
        0.015,
        "Shading: ±1 SD across training seeds (not a confidence interval).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(
        output_dir / "control_learning_curves.png",
        dpi=200
    )
    plt.close(fig)



def plot_learning_rate_sensitivity(
    method: str,
    learning_rates: list[float],
    checkpoints: np.ndarray,
    learning_curves: list[np.ndarray],
    output_dir: Path,
    n_eval_games: int,
    validation_seed: int,
    default_curve_mean: float | None = None
) -> None:
    """Plot one method across rates using precomputed per-seed wealth curves."""

    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = len(learning_curves[0])

    fig, ax = plt.subplots(figsize=(9, 5))

    for alpha_0, curves in zip(learning_rates, learning_curves):
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0)

        line, = ax.plot(checkpoints, mean, label=f"alpha0 = {alpha_0:g}")
        ax.fill_between(
            checkpoints, mean - sd, mean + sd,
            color=line.get_color(), alpha=0.2
        )

    if default_curve_mean is not None:
        ax.axhline(default_curve_mean, linestyle="--", color="grey",
                   label="Default vs default (checkpoint validation)")
    ax.axhline(100.0, linestyle=":", color="black", label="Hold cash = 100")

    ax.set(
        xlabel="Training episodes",
        ylabel="Mean validation terminal wealth",
        title=f"{method}: learning-rate sensitivity\n"
              f"{n_eval_games} validation games per checkpoint (seed {validation_seed})"
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


def plot_prediction_comparison(
    checkpoints: np.ndarray,
    mc_runs: np.ndarray,
    td_runs: np.ndarray,
    output_dir: str | Path
) -> None:
    """Plot MC/TD validation RMSE curves, supplied as (seeds, checkpoints)."""

    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(output_dir / "prediction_comparison.png", dpi=200)
    plt.close(fig)



def plot_prediction_learning_rates(
    steps: np.ndarray,
    schedules: dict[str, list[float]],
    output_dir: str | Path
) -> None:
    """Plot precomputed learning-rate schedules; no schedule logic lives here."""

    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the separate learning-rate schedules used for MC and TD.
    for name, values in schedules.items():
        ax.plot(steps, values, label=name)

    ax.set(
        xlabel="Training episode",
        ylabel="Step size alpha",
        title="Decreasing learning-rate schedules"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "prediction_learning_rate.png", dpi=200)
    plt.close(fig)

