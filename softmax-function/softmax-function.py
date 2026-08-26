import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    X = np.asarray(x)
    if len(X.shape) == 1:
        m = np.max(X)
        num = np.exp(X - m)
        denom = np.sum(num)
    elif len(X.shape) == 2:
        m = np.max(X, axis=1, keepdims=True)
        num = np.exp(X - m)
        denom = np.sum(num, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported input shape {X.shape}")
        
    return num / denom
