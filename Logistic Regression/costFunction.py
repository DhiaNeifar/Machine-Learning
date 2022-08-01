import numpy as np
from math import log, exp


from sigmoid import sigmoid


def costFunction(t, x, y) -> tuple:
    m = x.shape[0]
    one = np.ones((m, 1))
    H = sigmoid(np.dot(x, t))
    log_H = np.log(H)
    log_1_H = np.log(one - H)
    J = -np.sum(y * log_H + (one - y) * log_1_H) / m
    return J, np.dot(x.T, H - y) / m


def sig(z) -> float:
    return 1 / (1 + exp(-1 * z))


def cost(t, x, y) -> float:
    m, n = x.shape
    l = []
    for i in range(m):
        h = 0
        for j in range(n):
            h += t[j] * x[i, j]
        H = sig(h)
        if H != 0 and H != 1:
            l.append(y[i, 0] * log(H) + (1 - y[i, 0]) * log(1 - H))

    return -sum(l) / m


def cost_function_reg(t, x, y, l) -> tuple:
    m = x.shape[0]
    one = np.ones((m, 1))
    H = sigmoid(np.dot(x, t))
    log_H = np.log(H)
    log_1_H = np.log(one - H)
    J = -np.sum(y * log_H + (one - y) * log_1_H)
    reg_t = t.copy()
    reg_t[0, 0] = 0
    J += (np.dot(reg_t.T, reg_t)[0, 0] * l) / 2
    return J / m, (np.dot(x.T, H - y) + reg_t * l) / m


def cost_reg(t, x, y, lamda) -> float:
    m, n = x.shape
    l = []
    for i in range(m):
        h = 0
        for j in range(n):
            h += t[j] * x[i, j]
        H = sig(h)
        if H != 0 and H != 1:
            l.append(y[i, 0] * log(H) + (1 - y[i, 0]) * log(1 - H))
    J = -sum(l)
    reg_t = np.array(t).reshape(n, 1)
    reg_t[0, 0] = 0
    J += (np.dot(reg_t.T, reg_t)[0, 0] * lamda) / 2
    return J / m
