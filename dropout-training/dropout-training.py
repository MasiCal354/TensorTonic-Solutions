import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x, dtype=float)
    
    if p == 1.0:
        mask = np.zeros(x.shape, dtype=bool)
        return np.zeros_like(x), mask

    r = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    mask = r >= p
    scale = 1.0 / (1.0 - p)
    output = x * mask * scale
    
    return output, mask * scale
