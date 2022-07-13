import numpy as np
import pandas as pd


from FeatureScaling.FeatureScaling import featureScaling
from GradientDescent.GradientDescent import gradient_descent
from GradientDescent.GradientDescentMulti import gradient_descent_multi
from PlotData.plotdata import plot_convergence


def ex1_multi():

    """
        Machine Learning Online Class - Exercise 1: Linear Regression With Multiple Variables
    """

    # ==================== Part 1: Feature Normalization ====================

    print("Loading data ... ")
    col_names = ['house_size', 'br', 'price']
    data = pd.read_csv('ex1data2.txt', names=col_names, header=None, delimiter=',')

    X = data[['house_size', 'br']]
    y = data[['price']]

    print('First 10 examples from the dataset: ')

    # print(X.head(10))
    # print(y.head(10))

    print('Program paused. Press enter to continue.\n')
    input()

    print('Normalizing Features ...')
    X = featureScaling(X)
    X['bias'] = [1 for _ in range(len(X))]
    X = X[['bias', 'house_size', 'br']]

    # ================ Part 2: Gradient Descent ================

    print('Running gradient descent ...')
    alpha = 0.01
    num_iter = 400

    theta = np.zeros((3, 1))
    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iter)
    # print(J_history)
    plot_convergence(J_history)














if __name__ == '__main__':
    ex1_multi()
