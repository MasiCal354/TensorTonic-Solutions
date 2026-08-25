import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    S = np.asarray(scores)
    M = S.copy()
    upper_tri_mask = np.triu(np.ones(S.shape[-2:], dtype=bool), k=1)
    
    M[..., upper_tri_mask] = mask_value
    return M
