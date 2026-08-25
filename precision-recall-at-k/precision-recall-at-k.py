def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    num = len(set(recommended[:k]) & set(relevant))
    return [num / k, num / len(relevant)]
