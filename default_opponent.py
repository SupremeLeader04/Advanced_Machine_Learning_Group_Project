import numpy as np


def default_opponent_policy(state: list) -> list:
    """
    A default opponent policy that invests in speed and trading volume based on the current state.

    Args:
        state (list): The current state of the game.

    Returns:
        list: The policy output containing speed investment and trading volume.
    """
    
    B = float(state[1])

    # construct default speed investment:
    Z = np.random.uniform(0, 1)
    delta = float(Z * np.sqrt(B))

    # construct default trading volume:
    W = B - delta**2
    x = float(W * np.clip(state[5]/2, -1, 1))

    # Keep the action feasible under the simulator's strict floating-point check.
    while delta**2 > B:
        delta = float(np.nextafter(delta, 0.0))

    while delta**2 + abs(x) > B:
        x = float(np.nextafter(x, 0.0))

    return [delta, x]