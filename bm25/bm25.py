import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    N = len(docs)
    if N == 0:
        return np.array([], dtype=float)

    unique_query_terms = set(query_tokens)

    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = np.mean(doc_lengths)

    df_counter = Counter()
    for doc in docs:
        df_counter.update(set(doc))

    doc_counters = [Counter(doc) for doc in docs]

    scores = np.zeros(N, dtype=float)

    for term in unique_query_terms:
        df_t = df_counter[term]

        if df_t == 0:
            continue

        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)

        tf = np.array([doc_count[term] for doc_count in doc_counters], dtype=float)

        denom = tf + k1 * (1.0 - b + b * (doc_lengths / avgdl))

        term_scores = idf * (tf * (k1 + 1.0)) / denom
        scores += term_scores

    return scores
