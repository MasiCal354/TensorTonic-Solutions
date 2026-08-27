import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    X = np.asarray(matrix, dtype=float)
    
    if norm_type == "l1":
        norm = np.sum(np.abs(X), axis=axis, keepdims=True)
    elif norm_type == "l2":
        norm = np.sqrt(np.sum(X**2, axis=axis, keepdims=True))
    elif norm_type == "max":
        norm = np.max(np.abs(X), axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported norm_type: '{norm_type}'. Choose from 'l1', 'l2', or 'max'.")

    norm = np.where(norm == 0, 1.0, norm)
    
    return X / norm
