import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    token_index = {token: index for index, token in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=int)
    for token in tokens:
        if token in token_index:
            counts[token_index[token]] += 1
    return counts
