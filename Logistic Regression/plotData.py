import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def plotData(data) -> None:
    pos = data[data['admission'] == 1]
    neg = data[data['admission'] == 0]

    fig = plt.figure()
    plt.plot(pos['exam_1_score'], pos['exam_2_score'], 'b+', label='Admitted')
    plt.plot(neg['exam_1_score'], neg['exam_2_score'], 'yo', label='Not Admitted')
    plt.xlabel('exam_1_score')
    plt.ylabel('exam_2_score')
    plt.legend()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)


def plotData_reg(data) -> None:
    pos = data[data['admission'] == 1]
    neg = data[data['admission'] == 0]

    fig = plt.figure()
    plt.plot(pos['test_1'], pos['test_2'], 'k+', label='Accepted')
    plt.plot(neg['test_1'], neg['test_2'], 'yo', label='Rejected')
    plt.xlabel('test_1')
    plt.ylabel('test_2')
    plt.legend()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)


def plotDecisionBoundary(data, theta) -> None:
    pos = data[data['admission'] == 1]
    neg = data[data['admission'] == 0]
    x_min = data['exam_1_score'].min()
    x_max = data['exam_1_score'].max()
    y_min = find_opposite(x_min, theta)
    y_max = find_opposite(x_max, theta)

    fig = plt.figure()
    plt.plot(pos['exam_1_score'], pos['exam_2_score'], 'b+', label='Admitted')
    plt.plot(neg['exam_1_score'], neg['exam_2_score'], 'yo', label='Not Admitted')
    plt.plot([x_min, x_max], [y_min, y_max], marker='x')
    plt.xlabel('exam_1_score')
    plt.ylabel('exam_2_score')
    plt.legend()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)


def find_opposite(var, theta):
    return -(theta[0, 0] + var * theta[1, 0]) / theta[2, 0]


def plotDecisionBoundary_reg(data, theta) -> None:
    pos = data[data['admission'] == 1]
    neg = data[data['admission'] == 0]
    x_min, x_max = data['test_1'].min(), data['test_1'].max()
    y_min, y_max = data['test_2'].min(), data['test_2'].max()
    X = np.linspace(x_min, x_max, 100)
    Y = find_opposite_reg(X, theta, (y_min, y_max))
    print(Y.shape)
    fig = plt.figure()
    plt.plot(pos['test_1'], pos['test_2'], 'b+', label='Admitted')
    plt.plot(neg['test_1'], neg['test_2'], 'yo', label='Not Admitted')
    # plt.plot([X, Y], marker='x')
    plt.xlabel('test_1')
    plt.ylabel('test_2')
    plt.legend()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)


def find_opposite_reg(lin, theta, tup) -> np.ndarray:
    print(tup)
    d = {i: 0 for i in range(7)}
    y = np.zeros(lin.shape)
    for elt in range(100):
        ind = 0
        for i in range(6 + 1):
            for j in range(i + 1):
                d[j] += pow(lin[elt], j - i) * theta[ind]
                ind += 1
        bound = (-1, 1.1)
        bnds = bound
        result = opt.minimize(calc, x0=np.array([0]), args=d, method='SLSQP', bounds=bnds)
        y[elt] = result.x
    return y


def calc(x2, dic):
    return sum([(x2 ** i) * dic[i] for i in dic]) - dic[0] * 2


if __name__ == '__main__':
    x = np.linspace(1, 2, 100)
    print(find_opposite_reg(x, [0 for _ in range(100)]))
