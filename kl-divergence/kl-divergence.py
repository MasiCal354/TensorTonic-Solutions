import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    q = np.clip(q, eps, 1)
    return float(np.sum(np.where(p == 0, 0.0, p * np.log(p / q))))
