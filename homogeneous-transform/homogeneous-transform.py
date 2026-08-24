import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to one point or a batch.
    """
    T_arr = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float)

    R = T_arr[:3, :3]
    t = T_arr[:3, 3]

    return pts @ R.T + t