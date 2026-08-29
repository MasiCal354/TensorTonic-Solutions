def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    ema = []
    for v in values:
        if len(ema) == 0:
            ema.append(v)
        else:
            ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema
