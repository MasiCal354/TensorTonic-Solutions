import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the contrastive loss as a float.
    """
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)

    D = np.linalg.norm(A - B, axis=-1)

    L = Y * (D ** 2) + (1 - Y) * (np.maximum(0.0, margin - D) ** 2)
    
    if reduction == "mean":
        return float(np.mean(L))
    elif reduction == "sum":
        return float(np.sum(L))
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")
