import numpy as np

def confusion_matrix_norm(y_true: list, y_pred: list, num_classes: int | None = None, normalize: str = "none") -> np.ndarray:
    """
    Returns the confusion matrix as a NumPy array.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if num_classes is None:
        num_classes = 0 if y_true.size == 0 else int(max(y_true.max(), y_pred.max())) + 1

    matrix = np.bincount(
        y_true * num_classes + y_pred, minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    if normalize == "none":
        return matrix
    matrix = matrix.astype(float)

    if normalize == "true":
        totals = matrix.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        totals = matrix.sum(axis=0, keepdims=True)
    else:
        totals = np.array([[matrix.sum()]])

    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)
