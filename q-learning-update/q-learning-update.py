import numpy as np

def q_learning_update(Q: list, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as Q.
    """
    table = np.asarray(Q, dtype=float).copy()
    target = r + gamma * np.max(table[s_next])
    table[s, a] += alpha * (target - table[s, a])
    return table
