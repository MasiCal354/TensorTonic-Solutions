import numpy as np

def mean_average_precision(y_true_list: list, y_score_list: list, k: int | None = None) -> dict:
    """
    Returns a dictionary with map_value and ap_per_query.
    """
    ap_values = []

    for labels, scores in zip(y_true_list, y_score_list):
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)
        relevant = int(labels.sum())
        if relevant == 0:
            ap_values.append(0.0)
            continue
        ranked = labels[np.argsort(-scores, kind="stable")]
        limit = ranked.size if k is None else min(k, ranked.size)
        ranked = ranked[:limit]
        precision = np.cumsum(ranked) / np.arange(1, limit + 1)
        ap_values.append(float(np.sum(precision * ranked) / relevant))

    rounded = [round(value, 6) for value in ap_values]
    mean_value = float(np.mean(ap_values)) if ap_values else 0.0

    return {"map_value": round(mean_value, 6), "ap_per_query": rounded}
