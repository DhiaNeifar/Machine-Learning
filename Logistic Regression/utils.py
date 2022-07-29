import numpy as np


def n_decimal_places(X, n) -> float:
    d = pow(10, n)
    return int(X * d) / d


def n_decimal_numarray(x, p) -> np.ndarray:
    m, n = x.shape
    for i in range(m):
        for j in range(n):
            x[i, j] = n_decimal_places(x[i, j], p)
    return x
