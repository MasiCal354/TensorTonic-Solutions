import numpy as np
import math

def cyclic_encoding(values: list, period: float) -> list:
    """
    Returns the sine and cosine encoding of every cyclic value.
    """
    values = np.asarray(values)
    period = np.asarray(period)
    theta = 2 * np.pi * values / period
    
    return [[math.sin(t), math.cos(t)] for t in theta]
