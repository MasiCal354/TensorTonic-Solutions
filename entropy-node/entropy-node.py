import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    labels, counts = np.unique(y, return_counts=True)
    probs = counts/len(y)
    return float(-sum(probs * np.log2(probs, where=(probs != 0))))