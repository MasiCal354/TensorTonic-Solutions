import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.asarray(x)
    xbar = np.mean(x)
    s = np.sqrt(1 / (x.size - 1) * np.sum((x - xbar)**2))
    if s == 0.0:
        if xbar == mu0:
            return 0.0
        else:
            return float("inf")
    t = (xbar - mu0) / (s / np.sqrt(x.size))
    return float(t)
