import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)

    return float(
        np.mean(
            np.maximum(
                0,
                np.sum((anchor - positive)**2, axis=-1) - np.sum((anchor - negative)**2, axis=-1) + margin
            )
        )
    )
