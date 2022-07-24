import numpy as np


def sigmoid(m):
    z = np.ones(m.shape)
    return np.divide(z, np.add(z, np.exp(-m)))
