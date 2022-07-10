import numpy as np

from WarmUp.WarmUpAssignment import warmup
from PlotData.plotdata import plot_data


def ex1():

    """
        Machine Learning Online Class - Exercise 1: Linear Regression
    """

    # ==================== Part 1: Basic Function ====================

    print('Running warmUp exercise ... ')
    print('5x5 Identity Matrix:')
    print(warmup(4))
    print('Program paused. Press enter to continue.')
    input()

    # ======================= Part 2: Plotting =======================

    print('Plotting Data ...')
    data = np.loadtxt('ex1data1.txt', delimiter=',')

    X, y = data[:, 0], data[:, 1]   # X: population of a city, y: profit of a food truck in that city
    m = X.shape[0]  # m is the number of training examples

    plot_data(X, y)
    print('Program paused. Press enter to continue.')
    input()

    # =================== Part 3: Cost and Gradient descent ===================




if __name__ == '__main__':
    ex1()
