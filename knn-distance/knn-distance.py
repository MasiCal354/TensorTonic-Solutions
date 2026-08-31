import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    train = np.asarray(X_train, dtype=float)
    test = np.asarray(X_test, dtype=float)
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)
    distances = np.sqrt(np.sum((test[:, None, :] - train[None, :, :]) ** 2, axis=2))
    count = min(k, train.shape[0])
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :count]
    if count < k:
        padding = np.full((test.shape[0], k - count), -1, dtype=int)
        neighbors = np.concatenate((neighbors, padding), axis=1)
    return neighbors.astype(int)
