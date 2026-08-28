import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    matrix = np.asarray(A, dtype=float)
    size = matrix.shape[0]
    augmented = np.concatenate((matrix.copy(), np.eye(size)), axis=1)
    for column in range(size):
        pivot = column + np.argmax(np.abs(augmented[column:, column]))
        if abs(augmented[pivot, column]) < 1e-12:
            return None
        augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row in range(size):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]
    return augmented[:, size:]
