import numpy as np
import pandas as pd


from FeatureScaling.FeatureScaling import featureScaling
from GradientDescent.GradientDescent import gradient_descent
from GradientDescent.GradientDescentMulti import gradient_descent_multi
from PlotData.plotdata import plot_convergence, plot_multi_convergence
from normalEqn import normalEqn


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
    X, mu, sigma = featureScaling(X)
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

    print('Theta computed from gradient descent: ')
    print(f'{theta[0, 0]} \n{theta[1, 0]}\n{theta[2, 0]}')

    print('Computing for different learning_rates: ')

    multi_alpha = [0.3, 0.1, 0.03, 0.01]
    J_hist = []
    colors = ['b', 'g', 'r', 'c']
    for al in multi_alpha:
        _, j = gradient_descent_multi(X, y, theta, al, num_iter)
        J_hist.append(j)
    plot_multi_convergence(J_hist, colors, multi_alpha)

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ')
    house_size, br = 2104, 3
    x = np.array([1, (house_size - mu[0]) / sigma[0], (br - mu[1]) / sigma[1]])
    print(np.dot(x, theta))

    print('Theta computed from the normal equations: ')
    theta_nrm = normalEqn(X, y)
    print(f'{theta_nrm[0, 0]} \n{theta_nrm[1, 0]}\n{theta_nrm[2, 0]}')

    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', np.dot(x, theta_nrm))


if __name__ == '__main__':
    ex1_multi()
