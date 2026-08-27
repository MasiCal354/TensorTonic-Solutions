import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    if Z1.ndim == 1:
        Z1 = Z1[np.newaxis, :]
        Z2 = Z2[np.newaxis, :]

    S = np.dot(Z1, Z2.T) / temperature

    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)

    sum_exp_S = np.sum(exp_S, axis=1)

    exp_positive = exp_S.diagonal()

    log_probs = np.log(exp_positive / sum_exp_S)

    loss = -np.mean(log_probs)

    return float(loss)
