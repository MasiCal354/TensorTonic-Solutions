import numpy as np

def make_diagonal(v: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, N).
    """
    D = np.zeros((len(v), len(v)))
    np.fill_diagonal(D, v)
    return D
