import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    return float(np.mean(fake_scores) - np.mean(real_scores))
