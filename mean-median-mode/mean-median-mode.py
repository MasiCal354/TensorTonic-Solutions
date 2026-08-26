from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(max(Counter(x).items(), key=lambda pair: (pair[1], -pair[0]))[0]),
    }