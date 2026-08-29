from collections import Counter
def cohens_kappa(rater1: list, rater2: list) -> float:
    """
    Returns Cohen's kappa as a float.
    """
    n = len(rater1)
    po = sum(1 for i1, i2 in zip(rater1, rater2) if i1 == i2) / n
    C1 = Counter(rater1)
    C2 = Counter(rater2)
    keys = set(C1.keys()) | set(C2.keys())
    pe = sum(C1.get(key, 0) * C2.get(key, 0) for key in keys)/n**2
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)
