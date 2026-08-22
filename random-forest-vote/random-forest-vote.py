import numpy as np
from collections import defaultdict

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    output = []
    for p in zip(*predictions):
        votes = defaultdict(int)
        for v in p:
            votes[v] += 1
        output.append(max(votes.items(), key=lambda x: (x[1], -x[0]))[0])
    return output
