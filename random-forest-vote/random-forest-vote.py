import numpy as np
from scipy import stats

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    arr = np.asarray(predictions)
    mode_result = stats.mode(arr, axis=0, keepdims=False)
    return mode_result.mode.tolist()
