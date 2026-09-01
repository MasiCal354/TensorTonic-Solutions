import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as V.
    """
    values = np.asarray(V, dtype=float).copy()
    target = r + gamma * values[s_next]
    values[s] += alpha * (target - values[s])
    return values
