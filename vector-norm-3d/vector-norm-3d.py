import numpy as np

def vector_norm_3d(v: list) -> float | np.ndarray:
    """
    Returns a float or a NumPy array.
    """
    v = np.asarray(v)
    return np.sqrt(np.sum(v**2, axis=-1))
