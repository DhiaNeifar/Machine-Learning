import numpy as np

from FeatureScaling.FeatureScaling import featureScaling


def ex1_mutli():

    """
        Machine Learning Online Class - Exercise 1: Linear Regression With Multiple Variables
    """

    # ==================== Part 1: Feature Normalization ====================

    print("Loading data ... ")
    data = np.loadtxt('ex1data2.txt', delimiter=',')

    X1 = np.array([data[:, 0]]).T    # size of the house.
    X2 = np.array([data[:, 1]]).T    # number of bedrooms.
    X = np.concatenate((X1, X2), axis=1, dtype=float)
    y = np.array([data[:, 2]], dtype=float).T     # y: price of the house.
    m = len(y)                                # m is the number of training examples

    print('First 10 examples from the dataset: ')
    print(f'X = \n{X[:, :10]}')
    print(f'y = \n{y[:, :10]}')
    print('Program paused. Press enter to continue.\n')
    input()

    print('Normalizing Features ...')

    featureScaling(np.random.rand(5, 3))
    # % Add intercept term to X
    # X = [ones(m, 1) X];
















if __name__ == '__main__':
    ex1_mutli()
