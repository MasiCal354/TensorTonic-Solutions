import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    erf = np.vectorize(math.erf)
    X = np.asarray(x, dtype=np.float64)
    return X/2 * (1 + erf(X/2**0.5))
