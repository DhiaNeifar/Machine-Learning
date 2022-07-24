import numpy as np


def compute_Cost(X, y, theta) -> float:
    m = X.shape[0]
    result = np.dot(X, theta)
    result = np.add(result, -y)
    result = result * result
    return np.sum(result) / (2 * m)
