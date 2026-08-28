import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    x = np.asarray(x)
    arr = np.sort(x)
    q = np.asarray(q)
    r = q / 100.0 * (x.size - 1)
    l = np.floor(r).astype(int)
    u = np.ceil(r).astype(int)
    w = r - l
    pq = (1 - w) * arr[l] + w * arr[u]
    return pq
