import numpy as np

from ComputeCost.computeCost import compute_Cost


def gradient_descent(X, y, theta, alpha, num_iter) -> np.ndarray:
    m = X.shape[0]
    J_history = np.zeros((num_iter + 1, 1))

    for i in range(num_iter):
        theta = theta - (alpha / m) * np.dot(X.T, np.add(np.dot(X, theta), -y))
        J_history[num_iter] = compute_Cost(X, y, theta)

    return theta
