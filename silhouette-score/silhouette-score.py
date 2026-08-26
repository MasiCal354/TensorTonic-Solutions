import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    n_samples = X.shape[0]

    diffs = X[:, None, :] - X[None, :, :]
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=-1))

    same_cluster = labels[:, None] == labels[None, :]

    cluster_sizes = np.sum(same_cluster, axis=1, keepdims=True)
    a = np.sum(dist_matrix * same_cluster, axis=1) / (cluster_sizes.squeeze() - 1)

    unique_labels = np.unique(labels)
    b = np.full(n_samples, np.inf)
    
    for k in unique_labels:
        mask_k = (labels == k)

        mean_dist_k = np.sum(dist_matrix[:, mask_k], axis=1) / np.sum(mask_k)

        mean_dist_k[mask_k] = np.inf

        b = np.minimum(b, mean_dist_k)

    s = (b - a) / np.maximum(a, b)

    return float(np.mean(s))