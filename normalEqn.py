import numpy as np


def normalEqn(x, y):
    x_t = x.T
    H = np.linalg.inv(np.dot(x_t, x))
    return np.dot(np.dot(H, x_t), y)
