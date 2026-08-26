import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    Y = np.asarray(y_true)
    S = np.asarray(y_score)
    L = np.maximum(0, margin - Y * S)
    return float(np.mean(L)) if reduction == "mean" else float(np.sum(L))
