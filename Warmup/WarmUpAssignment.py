import numpy as np


def warmup(dim):
    mat = np.identity(dim)
    return mat


if __name__ == '__main__':
    matrix_dim = 5
    matrix = warmup(matrix_dim)
    print(matrix)
