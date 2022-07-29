import numpy as np


from sigmoid import sigmoid


def predict(t, x):
    m, _ = x.shape
    z = np.ones((m, 1)) * 0.5
    return np.greater(sigmoid(np.dot(x, t)), z, where=1)
