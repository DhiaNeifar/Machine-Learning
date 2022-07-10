import numpy as np

from WarmUp.WarmUpAssignment import warmup
from PlotData.plotdata import plot_data
from PlotData.plotdata import plot_data_1
from ComputeCost.computeCost import compute_Cost
from GradientDescent.GradientDescent import gradient_descent
from utils import n_decimal_places


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

    X = np.array([data[:, 0]], dtype=float).T    # X: population of a city
    y = np.array([data[:, 1]], dtype=float).T    # y: profit of a food truck in that city
    m = X.shape[0]  # m is the number of training examples

    plot_data(X, y)
    print('Program paused. Press enter to continue.')
    input()

    # =================== Part 3: Cost and Gradient descent ===================

    X = np.concatenate((np.ones((m, 1)), X), axis=1, dtype=float)

    theta = np.zeros((2, 1), dtype=float)

    iterations = 1500
    alpha = 0.01

    print('Testing the cost function ...')

    J = compute_Cost(X, y, theta)
    print(f'With theta = [0, 0]\nCost computed = {n_decimal_places(J, 2)}')
    print('Expected cost value (approx) 32.07')

    J = compute_Cost(X, y, np.array([[-1, 2]], dtype=float).T)

    print('With theta = [-1, 2]\nCost computed = {:.2f}'.format(J))
    print('Expected cost value (approx) 54.24')

    print('Program paused. Press enter to continue.')
    input()

    print('Running Gradient Descent ...')

    theta = gradient_descent(X, y, theta, alpha, iterations)
    print(f'Theta found by gradient descent: {n_decimal_places(theta[0, 0], 4)} {n_decimal_places(theta[1, 0], 4)}')
    print('Expected theta values (approx) -3.6303 1.1664')

    plot_data_1(X, y, theta)

    print('Predicting values for population sizes of 35, 000 and 70, 000')

    predict1 = np.dot(np.array([[1, 3.5]], dtype=float), theta)
    print(f'For population = 35,000, we predict a profit of {predict1[0, 0] * 10000}')

    predict2 = np.dot(np.array([[1, 7]], dtype=float), theta)
    print(f'For population = 70,000, we predict a profit of {predict2[0, 0] * 10000}')

    print('Program paused. Press enter to continue.\n')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============

    
if __name__ == '__main__':
    ex1()
