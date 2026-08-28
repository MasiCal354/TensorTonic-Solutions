import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    return np.sum(np.diag(A))
