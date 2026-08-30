import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    keys = train_dist.keys()
    output = {}
    for key in keys:
        train = np.asarray(train_dist[key]) + eps
        serving = np.asarray(serving_dist[key]) + eps
        psi = np.sum((serving - train) * np.log(serving / train))
        if psi >= threshold:
            skewed = True
        else:
            skewed = False
        output[key] = {"psi": psi, "skewed": skewed}
    return output
