import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    return {
        "pmf": float(math.comb(n, k) * p**k * (1 - p)**(n - k)),
        "cdf": float(sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1)))
    }
