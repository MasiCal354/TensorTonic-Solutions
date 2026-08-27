import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    G = np.asarray(g)
    Gnorm = np.sqrt(np.sum(G**2))
    Gclipped = np.where(Gnorm <= max_norm, G, G * max_norm/Gnorm)
    return Gclipped
