def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    sumpa = sum(p**alpha for p in priorities)
    P = [p**alpha / sumpa for p in priorities]
    w = [(len(priorities) * Pi)**-beta for Pi in P]
    maxw = max(w)
    return [P, [wi/maxw for wi in w]]
