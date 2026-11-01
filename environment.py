from math import exp, isfinite
from numbers import Real
from types import FunctionType

import numpy as np

from default_opponent import default_opponent_policy


def softmax_probabilities(S_1, S_2) -> list:
    """
    Computes the probabilities of player 1 and player 2 winning the speed lottery using a softmax function.

    Args:
        S_1 (float): Effective speed of player 1.
        S_2 (float): Effective speed of player 2.

    Returns:
        list[float]: Probabilities of player 1 and player 2 winning the speed lottery.
    """
    exp_S_1 = exp(2*S_1)
    exp_S_2 = exp(2*S_2)
    total = exp_S_1 + exp_S_2
    return [exp_S_1 / total, exp_S_2 / total]

def check_policy(policy: list, state: list) -> bool:
    """
    Checks whether a policy output has the expected structure.

    Args:
        policy (list): The policy output to check.
        state (list): The current state of the game.

    Returns:
        bool: True if the policy output is valid.

    Raises:
        TypeError: If the policy output is not a list.
        ValueError: If the policy output does not contain two finite numbers.
    """
    if not isinstance(policy, list):
        raise TypeError("Policy output must be a list containing speed investment and trading volume.")
    if len(policy) != 2:
        raise ValueError("Policy output must contain exactly 2 values: speed investment and trading volume.")
    for value, label in zip(policy, ("speed investment", "trading volume")):
        if not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"Policy {label} must be a finite number; got {value!r}.")
    if policy[0] < 0:
        raise ValueError("Policy speed investment must be non-negative.")
    if policy[0]**2 + abs(policy[1]) > state[1]:
        raise ValueError(
            "Policy speed investment cost and absolute trading volume "
            "must not exceed the available budget."
        )
    return True


def sim_game(
    s: list | None = None,
    n: int = 10,
    policy_1: FunctionType = default_opponent_policy,
    policy_2: FunctionType = default_opponent_policy,
) -> np.ndarray:
    """
    Simulates a game between two players using their respective policies.

    Args:
        s (list | None): The initial state of the game. Defaults to
            [0, 100, 100, 0, 0, 0, 0].
        n (int): The number of turns to simulate. Defaults to 10.
        policy_1 (FunctionType): The policy function for player 1.
        policy_2 (FunctionType): The policy function for player 2.

    Returns:
        np.ndarray: A matrix with one row per period and a final terminal-state row.
    """
    
    if not callable(policy_1):
        raise TypeError("policy_1 must be a callable policy function.")
    if not callable(policy_2):
        raise TypeError("policy_2 must be a callable policy function.")

    if s is None:
        s = [0, 100, 100, 0, 0, 0, 0]

    # 1. check if initial state is reasonable:
    if not isinstance(s, list) or len(s) != 7:
        raise ValueError("Initial state must be a list with exactly 7 entries.")
    if any(not isinstance(value, Real) or not isfinite(value) for value in s):
        raise ValueError("Initial state entries must be finite numbers.")
    if s[0] != 0:
        raise ValueError("Initial state must have t == 0.")
    if s[1] != 100 or s[2] != 100:
        raise ValueError("Initial state must have both players with budget of 100.")
    if s[3] != 0 or s[4] != 0:
        raise ValueError("Initial state must have both players with speed capital of 0.")
    if s[6] < 0 or s[6] > 1:
        raise ValueError("Initial state must have a valid informativeness (between 0 and 1).")

    s = s.copy()

    # 2. simulation loop:
    if not isinstance(n, int) or n < 0:
        raise ValueError("Number of turns must be a non-negative integer.")
    records = []
    for turn in range(n):
        s[0] = turn

        # random draws of informativeness and signal:
        R_t: float = np.random.uniform(low=0.0, high=1.0)
        g_t: float = np.random.normal(loc=0.0, scale=R_t * 1.5)

        # update state:
        s[5] = g_t
        s[6] = R_t
        period_state = s.copy()

        # apply policy for player 1 and check feasibility:
        policy_state_1 = s.copy()
        pi_1: list = policy_1(policy_state_1)
        check_policy(pi_1, policy_state_1)

        # apply policy for player 2 and check feasibility:
        policy_state_2 = [s[0], s[2], s[1], s[4], s[3], s[5], s[6]]
        pi_2: list = policy_2(policy_state_2)
        check_policy(pi_2, policy_state_2)

        # applying speed investment costs to budgets:
        W_1_new = s[1] - pi_1[0]**2
        W_2_new = s[2] - pi_2[0]**2

        # determine winner of speed lottery:
        eff_speed_1 = s[3] + pi_1[0]
        eff_speed_2 = s[4] + pi_2[0]

        prob_1, prob_2 = softmax_probabilities(eff_speed_1, eff_speed_2)
        selection = np.random.choice([1, 2], p=[prob_1, prob_2])

        # forming move:
        epsilon_t = np.random.normal(loc=0.0, scale=1.5 * np.sqrt(1 - R_t**2))
        delta = g_t + epsilon_t

        # settle trade of winning firm:
        if selection == 1:
            s[1] = max(0, W_1_new + pi_1[1] * delta)
            s[2] = W_2_new
        else:
            s[1] = W_1_new
            s[2] = max(0, W_2_new + pi_2[1] * delta)

        # apply random speed decay:
        d_1 = (np.random.uniform(0, np.sqrt(0.1)))**2
        d_2 = (np.random.uniform(0, np.sqrt(0.1)))**2

        s[3] = eff_speed_1 * (1 - d_1)
        s[4] = eff_speed_2 * (1 - d_2)

        records.append(
            period_state
            + [
                epsilon_t,
                selection,
                pi_1[0],
                pi_1[1],
                pi_2[0],
                pi_2[1],
            ]
        )

    s[0] = n
    records.append(s.copy() + [np.nan] * 6)
    return np.array(records, dtype=float)