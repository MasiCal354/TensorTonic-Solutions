import numpy as np

def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    V = np.array(values, dtype=float)
    T = np.array(transitions, dtype=float)
    R = np.array(rewards, dtype=float)

    expected_next_values = T @ V

    Q = R + gamma * expected_next_values

    V_new = np.max(Q, axis=1)

    return V_new.tolist()
