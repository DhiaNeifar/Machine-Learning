import numpy as np

from ComputeCost.computeCost import compute_Cost


def gradient_descent_multi(X, Y, theta, alpha, num_iter):
    x = X.to_numpy()
    y = Y.to_numpy()
    m = X.shape[0]
    J_history = np.zeros((num_iter, 1))

    for i in range(num_iter):
        theta = theta - (alpha / m) * np.dot(x.T, np.add(np.dot(x, theta), -y))
        J_history[i] = compute_Cost(x, y, theta)
    return theta, J_history

