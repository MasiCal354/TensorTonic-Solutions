import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Returns a dictionary with X_train, X_test, y_train, and y_test.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    train_idx_list = []
    test_idx_list = []

    for label in classes:
        indices = np.flatnonzero(y == label)
        shuffled_indices = rng.permutation(indices)
        n_test = round(len(indices) * test_size)
        test_idx_list.append(shuffled_indices[:n_test])
        train_idx_list.append(shuffled_indices[n_test:])

    train_idx = np.sort(np.concatenate(train_idx_list))
    test_idx = np.sort(np.concatenate(test_idx_list))
    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx]
    }
