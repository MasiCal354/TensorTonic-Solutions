def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    xbar = sum(series) / len(series)
    gamma0 = sum((x - xbar)**2 for x in series)
    if gamma0 == 0:
        return [1.0] + [0.0] * max_lag

    return [
        sum(
            (series[t] - xbar) * (series[t + k] - xbar)
            for t in range(len(series) - k)
        ) / gamma0 for k in range(max_lag + 1)
    ]
