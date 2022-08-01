import numpy as np


def sigmoid(m) -> np.ndarray:
    z = np.ones(m.shape)
    return np.divide(z, np.add(z, np.exp(-m)))
