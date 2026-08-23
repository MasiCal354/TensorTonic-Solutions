import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """Return the training-time BatchNorm output."""
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 4:
        axis = (0, 2, 3)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    else:
        raise ValueError(f"Expected 2D or 4D input array, got shape {x.shape}")

    mu = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y
