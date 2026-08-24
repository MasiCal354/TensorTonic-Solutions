import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.array(X, dtype=float)
    n = X.shape[0]

    X_centered = X - np.mean(X, axis=0)

    C = (X_centered.T @ X_centered) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    idx = np.argsort(eigenvalues)[::-1]
    top_k_eigenvectors = eigenvectors[:, idx[:k]]

    for col in range(k):
        max_abs_idx = np.argmax(np.abs(top_k_eigenvectors[:, col]))
        if top_k_eigenvectors[max_abs_idx, col] < 0:
            top_k_eigenvectors[:, col] *= -1

    X_proj = X_centered @ top_k_eigenvectors

    return X_proj.tolist()
