import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    """
    Returns the state values as a NumPy array.
    """
    totals = np.zeros(n_states, dtype=float)
    counts = np.zeros(n_states, dtype=int)
    for episode in episodes:
        returns = np.zeros(len(episode), dtype=float)
        running = 0.0
        for index in range(len(episode) - 1, -1, -1):
            running = episode[index][1] + gamma * running
            returns[index] = running
        visited = set()
        for index, (state, _) in enumerate(episode):
            if state not in visited:
                totals[state] += returns[index]
                counts[state] += 1
                visited.add(state)
    values = np.zeros(n_states, dtype=float)
    np.divide(totals, counts, out=values, where=counts != 0)
    return np.round(values, 4)
