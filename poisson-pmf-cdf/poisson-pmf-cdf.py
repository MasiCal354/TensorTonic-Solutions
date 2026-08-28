import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    return {
        "pmf": float(math.exp(-lam) * lam**k / math.factorial(k)),
        "cdf": float(sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k + 1)))
    }
