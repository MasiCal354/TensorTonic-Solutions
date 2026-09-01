def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    out = []
    for i in range(0, len(X), pool_size):
        row = []
        for j in range(0, len(X[0]), pool_size):
            row.append(max(X[k][l] for k in range(i, i + pool_size) for l in range(j, j + pool_size)))
        out.append(row)
    return out
