import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.asarray(X)
    N = X.shape[0]
    mu = np.mean(X, axis=0)
    Xc = X - mu
    sigma = (np.transpose(Xc) @ Xc) / (N - 1)
    return sigma
