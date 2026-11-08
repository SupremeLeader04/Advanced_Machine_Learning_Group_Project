# 2.3.1: Feature Function

import numpy as np

# State feature functions 

def time_features(t: int) -> np.ndarray:
    """Encode periods 0–9 into five two-period one-hot bins."""

    if isinstance(t, (bool, np.bool_)) or not isinstance(
        t, (int, float, np.integer, np.floating)
    ):
        raise TypeError("Time period must be numeric.")

    if not np.isfinite(t) or not 0 <= t <= 9 or t != int(t):
        raise ValueError("Time period must be an integer between 0 and 9.")

    t = int(t)
    features = np.zeros(5, dtype=int)
    features[t // 2] = 1

    return features


def budget_features(B: float) -> np.ndarray:
    """
    Converts a firm's budget into eight binary indicators.

    Budget ranges: [0,20), [20,40), [40,60), [60,80),
    [80,100), [100,120), [120,150), [150,infinity).
    """

    if not isinstance(B, (int, float, np.integer, np.floating)):
        raise TypeError("Budget must be a number.")

    if not np.isfinite(B) or B < 0:
        raise ValueError("Budget must be finite and non-negative.")

    bin_edges = [20, 40, 60, 80, 100, 120, 150]

    # Values exactly on a boundary are placed in the higher budget range.
    bin_index = np.searchsorted(bin_edges, B, side="right")

    features = np.zeros(8, dtype=int)
    features[bin_index] = 1

    return features


def speed_features(S_self: float, S_opp: float) -> np.ndarray:
    """
    Converts the difference between the firms' speed capital
    into seven binary indicators.
    """

    for speed in (S_self, S_opp):
        if not isinstance(speed, (int, float, np.integer, np.floating)):
            raise TypeError("Speed capital must be numeric.")

        if not np.isfinite(speed) or speed < 0:
            raise ValueError("Speed capital must be finite and non-negative.")

    speed_diff = S_self - S_opp

    # Keep equal speeds in a separate category.
    if speed_diff < -5:
        bin_index = 0
    elif speed_diff < -3:
        bin_index = 1
    elif speed_diff < 0:
        bin_index = 2
    elif speed_diff == 0:
        bin_index = 3
    elif speed_diff < 3:
        bin_index = 4
    elif speed_diff < 5:
        bin_index = 5
    else:
        bin_index = 6

    features = np.zeros(7, dtype=int)
    features[bin_index] = 1

    return features


def informativeness_features(R: float) -> np.ndarray:
    """
    Converts signal informativeness (0 to 1) into ten
    binary indicators, each covering a range of 0.1.
    """

    if isinstance(R, (bool, np.bool_)) or not isinstance(
        R, (int, float, np.integer, np.floating)
    ):
        raise TypeError("Signal informativeness must be numeric.")

    if not np.isfinite(R) or not 0 <= R <= 1:
        raise ValueError("Signal informativeness must be between 0 and 1.")

    bin_edges = np.arange(1, 10) / 10

    # R = 1 is included in the final category.
    bin_index = np.searchsorted(bin_edges, R, side="right")

    features = np.zeros(10, dtype=int)
    features[bin_index] = 1

    return features


def signal_sign_features(g: float) -> np.ndarray:
    """
    Converts the trading signal's direction into three binary
    indicators: negative, zero or positive.
    """

    if isinstance(g, (bool, np.bool_)) or not isinstance(
        g, (int, float, np.integer, np.floating)
    ):
        raise TypeError("Trading signal must be numeric.")

    if not np.isfinite(g):
        raise ValueError("Trading signal must be finite.")

    if g < 0:
        bin_index = 0
    elif g == 0:
        bin_index = 1
    else:
        bin_index = 2

    features = np.zeros(3, dtype=int)
    features[bin_index] = 1

    return features


def signal_magnitude_features(g: float) -> np.ndarray:
    """
    Converts the absolute size of the trading signal into
    nine binary indicators, using narrower ranges for
    smaller signals.
    """

    if isinstance(g, (bool, np.bool_)) or not isinstance(
        g, (int, float, np.integer, np.floating)
    ):
        raise TypeError("Trading signal must be numeric.")

    if not np.isfinite(g):
        raise ValueError("Trading signal must be finite.")

    magnitude = abs(g)
    bin_edges = [0.5, 1, 1.5, 2, 2.5, 3, 4, 10]

    bin_index = np.searchsorted(bin_edges, magnitude, side="right")

    features = np.zeros(9, dtype=int)
    features[bin_index] = 1

    return features


# Combined state representation

def state_features(s: list) -> np.ndarray:
    """
    Combines the individual state variables into a
    50-dimensional vector of binary indicators.

    State: [t, B_self, B_opp, S_self, S_opp, g, R]
    """

    if not isinstance(s, (list, tuple, np.ndarray)) or len(s) != 7:
        raise ValueError("State must contain exactly 7 variables.")

    t, B_self, B_opp, S_self, S_opp, g, R = s

    features = [
        time_features(t),
        budget_features(B_self),
        budget_features(B_opp),
        speed_features(S_self, S_opp),
        informativeness_features(R),
        signal_sign_features(g),
        signal_magnitude_features(g)
    ]

    return np.concatenate(features)


# Generating feasible actions

def feasible_actions(s: list) -> list:
    """
    Generates feasible speed investments and trading positions.

    Both spending fractions range from 0% to 100% in 10% steps.
    Trading direction follows the signal, and total spending
    cannot exceed the firm's budget.
    """

    B = s[1]
    g = s[5]

    if not np.isfinite(B) or B < 0:
        raise ValueError("Budget must be finite and non-negative.")

    if not np.isfinite(g):
        raise ValueError("Trading signal must be finite.")

    if B == 0:
        return [(0.0, 0.0)]

    speed_fractions = np.linspace(0, 1, 11)

    # A zero signal only allows a zero trading position.
    trading_fractions = (
        [0.0] if g == 0 else np.linspace(0, 1, 11)
    )

    actions = []

    for speed_fraction in speed_fractions:

        # Speed investment costs delta squared.
        delta = float(np.sqrt(B * speed_fraction))

        # Avoid floating-point overspending under the environment strict constraint.
        while delta**2 > B:
            delta = float(np.nextafter(delta, 0.0))

        W = max(0.0, B - delta**2)

        # Spending the entire budget on speed leaves no trading position.
        if speed_fraction == 1:
            actions.append((delta, 0.0))
            continue

        for trading_fraction in trading_fractions:

            x = float(np.sign(g) * trading_fraction * W)

            # Correct small rounding errors that could make an action infeasible.
            while delta**2 + abs(x) > B:
                x = float(np.nextafter(x, 0.0))

            actions.append((delta, x))

    return actions


# State-action feature representation

def state_action_features(s: list, a: tuple) -> np.ndarray:
    """
    Constructs the 150-dimensional state-action feature vector.

    Combines 50 state indicators with
    the fraction spent on speed (delta²/B) and the fraction of
    remaining wealth allocated to trading (|x|/W).
    """

    phi_s = state_features(s)

    B = s[1]
    delta, x = a

    if not np.isfinite(delta) or not np.isfinite(x):
        raise ValueError("Action values must be finite.")

    if delta < 0 or delta**2 + abs(x) > B + 1e-10:
        raise ValueError("Action exceeds the available budget.")

    if B == 0:
        speed_fraction = 0.0
    else:
        speed_fraction = delta**2 / B

    W = max(0.0, B - delta**2)

    # Trading exposure is relative to the budget left after speed spending.
    if W == 0:
        trading_fraction = 0.0
    else:
        trading_fraction = abs(x) / W

    features = np.concatenate([
        phi_s,
        speed_fraction * phi_s,
        trading_fraction * phi_s
    ])

    return features




