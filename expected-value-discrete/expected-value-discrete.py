import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Return the expected value of the discrete distribution.
    """
    return float((np.asarray(x)*np.asarray(p)).sum().item())
