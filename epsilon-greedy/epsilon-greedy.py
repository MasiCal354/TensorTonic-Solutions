import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    values = np.asarray(q_values, dtype=float)
    rng = np.random.default_rng(seed)
    return int(rng.integers(values.size)) if rng.random() < epsilon else int(np.argmax(values))
