import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    e = y_true - y_pred
    return float(np.mean(np.where(np.abs(e) > delta, delta * (np.abs(e) - 0.5 * delta), 0.5 * e**2)))
