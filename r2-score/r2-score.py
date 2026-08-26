import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    Y = np.asarray(y_true)
    Yhat = np.asarray(y_pred)
    Ybar = np.mean(Y)
    denom = np.sum((Y - Ybar)**2)
    if denom == 0:
        return 1.0 if (Y == Yhat).all() else 0.0
    return float(1 - np.sum((Y - Yhat)**2)/denom)
