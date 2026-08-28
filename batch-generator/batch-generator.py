import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    n_samples = len(X)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > n_samples and drop_last:
            break
        yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]
