import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    X = np.asarray(x)
    n = X.size
    xbar = np.mean(X)
    var = np.sum((X - xbar)**2)/(n - 1)
    std = var**0.5
    return {
        "variance": float(var),
        "standard_deviation": float(std),
    }