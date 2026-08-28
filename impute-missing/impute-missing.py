import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    X_arr = np.array(X, dtype=float)
    X_imputed = X_arr.copy()
    original_shape = X_arr.shape

    if X_imputed.ndim == 1:
        X_imputed = X_imputed.reshape(-1, 1)

    for i in range(X_imputed.shape[1]):
        col = X_imputed[:, i]
        mask = np.isnan(col)
        observed = col[~mask]

        if len(observed) == 0:
            fill_value = 0.0
        elif strategy == 'mean':
            fill_value = np.mean(observed)
        else:
            fill_value = np.median(observed)

        col[mask] = fill_value

    if len(original_shape) == 1:
        X_imputed = X_imputed.reshape(original_shape)

    return X_imputed
