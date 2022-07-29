import matplotlib.pyplot as plt


def plotData1(data) -> None:
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


def find_opposite(x, theta):
    return -(theta[0, 0] + x * theta[1, 0]) / theta[2, 0]
