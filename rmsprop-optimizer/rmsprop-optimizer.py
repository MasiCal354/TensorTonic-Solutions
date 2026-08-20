import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Perform one RMSProp update step.
    """
    s = beta * np.array(s) + (1 - beta) * np.array(g)**2
    w = np.array(w) - np.array(g) * lr / ((s + eps)**0.5)
    return w, s
