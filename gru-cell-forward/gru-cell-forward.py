import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    x = np.asarray(x)
    h_prev = np.asarray(h_prev)
    Uh = np.asarray(params["Uh"])
    Ur = np.asarray(params["Ur"])
    Uz = np.asarray(params["Uz"])
    Wh = np.asarray(params["Wh"])
    Wr = np.asarray(params["Wr"])
    Wz = np.asarray(params["Wz"])
    bh = np.asarray(params["bh"])
    br = np.asarray(params["br"])
    bz = np.asarray(params["bz"])
    z = 1 / (1 + np.exp(-(x @ Wz + h_prev @ Uz + bz)))
    r = 1 / (1 + np.exp(-(x @ Wr + h_prev @ Ur + br)))
    h_tilde = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)
    h = (1 - z) * h_prev + z * h_tilde
    return h
