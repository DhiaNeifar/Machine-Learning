import matplotlib.pyplot as plt
import numpy as np


from mapfeature import map_feature_row


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

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((u.size, v.size))
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            z[i, j] = map_feature_row(ui, vj, theta)

    z = z.T
    fig = plt.figure()

    plt.plot(pos['test_1'], pos['test_2'], 'b+', label='Admitted')
    plt.plot(neg['test_1'], neg['test_2'], 'yo', label='Not Admitted')
    # plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
    plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)
    plt.xlabel('test_1')
    plt.ylabel('test_2')
    plt.legend()
    plt.waitforbuttonpress(0)
    plt.close(fig)
