import numpy as np

def normalize_3d(v: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as v.
    """
    values = np.asarray(v, dtype=float)
    norms = np.sqrt(np.sum(values ** 2, axis=-1, keepdims=True))
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)
