import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    return [i if i > 0 else alpha * (math.exp(i) - 1) for i in x]
