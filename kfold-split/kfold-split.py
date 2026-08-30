import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    indices = np.arange(N)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    folds = np.array_split(indices, k)
    result = []
    for index, validation in enumerate(folds):
        training = np.concatenate(folds[:index] + folds[index + 1:])
        result.append({"train_idx": training.astype(int), "val_idx": validation.astype(int)})
    return result
