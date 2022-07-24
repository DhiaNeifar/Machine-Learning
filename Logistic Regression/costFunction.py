import numpy as np
import pandas as pd
from math import log, exp


from sigmoid import sigmoid


file = '/home/dhianeifar/PycharmProjects/Machine_Learning/Logistic Regression/ex2data1.txt'


def costFunction(t, x, y):
    m = x.shape[0]
    H = sigmoid(np.dot(x, t))
    log_H = np.log(H)
    J = -np.sum(y * log_H + (np.ones((m, 1)) - y) * log_H) / m
    return J, np.dot(x.T, H - y) / m


def cost(t, x, y):
    print(x[: 10])
    # H = sigmoid(np.dot(x, t))
    # log_H = np.log(H)
    # J = -np.sum(y * log_H + (np.ones((m, 1)) - y) * log_H) / m
    # return J


def sig(z):
    return 1 / (1 + exp(-1 * z))


def f1(t, x, y):
    m, n = x.shape
    x_list = x.values.tolist()
    y_list = y.values.tolist()

    j = 0
    for i in range(m):
        h = 0
        for j in range(n):
            h += t[j] * x_list[i][j]
        H = sig(h)
        if H != 0 and H != 1:
            j -= y_list[i][0] * log(H) + (1 - y_list[i][0]) * log(1 - H)

    return j / m



def f(x):

    col_names = ['exam_1_score', 'exam_2_score', 'admission']
    data = pd.read_csv(file, names=col_names, header=None, delimiter=',')
    X = data[['exam_1_score', 'exam_2_score']]
    y = data[['admission']]
    X['bias'] = [1 for _ in range(len(X))]
    X = X[['bias', 'exam_1_score', 'exam_2_score']]
    m, n = X.shape
    print(m)
    H = sigmoid(np.dot(X, x))
    log_H = np.log(H)
    J = -np.sum(y * log_H + (np.ones((m, 1)) - y) * log_H) / m
    return J


if __name__ == '__main__':
    print(costFunction())

