import numpy as np


def default_opponent_policy(state: list) -> list:
    """
    A default opponent policy that invests in speed and trading volume based on the current state.

    Args:
        state (list): The current state of the game.

    Returns:
        list: The policy output containing speed investment and trading volume.
    """
    
    # construct default speed investment:
    Z = np.random.uniform(0, 1)
    delta = Z * np.sqrt(state[1])

    # construct default trading volume:
    W = state[1] - delta**2
    x = W * np.clip(state[5]/2, -1, 1)

    return [delta, x]