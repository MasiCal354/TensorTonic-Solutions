import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.asarray(X)
    Xmin = np.min(X, axis=axis, keepdims=True)
    Xmax = np.max(X, axis=axis, keepdims=True)
    data_range = Xmax - Xmin
    denom = np.where(data_range > eps, data_range, 1.0)
    return (X - Xmin)/denom
