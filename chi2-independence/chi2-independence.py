import numpy as np

def chi2_independence(C: list) -> dict:
    O = np.array(C, dtype=float)

    row_sums = O.sum(axis=1)
    col_sums = O.sum(axis=0)
    N = O.sum()

    E = np.outer(row_sums, col_sums) / N

    chi2 = np.sum((O - E) ** 2 / E)

    return {
        "chi2": float(chi2),
        "expected": E,
    }
