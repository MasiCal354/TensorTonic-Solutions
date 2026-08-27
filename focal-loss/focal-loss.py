import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """
    Returns the loss as a float.
    """
    p = np.asarray(p)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    y = np.asarray(y)
    L = -(1 - p)**gamma * y * np.log(p) - p**gamma * (1 - y) * np.log1p(-p)
    return float(np.mean(L))
