import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    X = np.asarray(x)
    return {
        "pmf": np.where(X == 1, p, 1.0 - p),
        "mean": float(p),
        "variance": float(p * (1 - p)),
    }
