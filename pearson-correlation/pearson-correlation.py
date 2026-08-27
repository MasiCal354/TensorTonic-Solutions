import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.asarray(X)
    N = X.shape[0]
    mu = np.mean(X, axis=0)
    Xc = X - mu
    sigma = (np.transpose(Xc) @ Xc) / (N - 1)
    std = np.sqrt(np.diag(sigma))
    denom = np.outer(std, std)
    return sigma / denom
