import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix (NumPy array) and vocabulary (sorted list).
    """
    N = len(documents)

    doc_tokens = [doc.lower().split() for doc in documents]

    vocab_set = set()
    for tokens in doc_tokens:
        vocab_set.update(tokens)
    vocabulary = sorted(list(vocab_set))

    vocab_to_idx = {term: idx for idx, term in enumerate(vocabulary)}

    df_counter = Counter()
    for tokens in doc_tokens:
        df_counter.update(set(tokens))

    idf = {}
    for term, df_val in df_counter.items():
        idf[term] = math.log(N / df_val)

    V = len(vocabulary)
    tfidf_matrix = np.zeros((N, V), dtype=float)

    for i, tokens in enumerate(doc_tokens):
        doc_len = len(tokens)
        if doc_len == 0:
            continue

        term_counts = Counter(tokens)

        for term, count in term_counts.items():
            col_idx = vocab_to_idx[term]
            tf = count / doc_len
            tfidf_matrix[i, col_idx] = tf * idf[term]

    return {
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary
    }
