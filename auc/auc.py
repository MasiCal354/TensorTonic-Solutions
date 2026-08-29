import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    lfpr = np.roll(fpr, shift=1)
    ltpr = np.roll(tpr, shift=1)
    arr = (fpr - lfpr) * (ltpr + tpr) / 2
    return float(np.sum(arr[1:]))
