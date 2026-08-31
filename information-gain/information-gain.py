import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    labels = np.asarray(y)
    mask = np.asarray(split_mask, dtype=bool)

    def entropy(values):
        if values.size == 0:
            return 0.0
        counts = np.unique(values, return_counts=True)[1]
        probabilities = counts / values.size
        return float(-np.sum(probabilities * np.log2(probabilities)))

    left = labels[mask]
    right = labels[~mask]
    if left.size == 0 or right.size == 0:
        return 0.0
    weighted = (left.size * entropy(left) + right.size * entropy(right)) / labels.size
    return float(entropy(labels) - weighted)
