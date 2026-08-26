import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X = np.asarray(X)
    mu = np.mean(X, axis=axis, keepdims=True)
    sigma = np.std(X, axis=axis, keepdims=True)
    denom = np.where(sigma > eps, sigma, 1.0)
    return (X - mu)/denom
