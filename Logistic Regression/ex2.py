import pandas as pd
import numpy as np
import scipy.optimize as opt


from plotData import plotData1
from costFunction import costFunction, f, f1, cost
from sigmoid import sigmoid
from utils import n_decimal_places


def ex2():

    """
        Machine Learning Online Class - Exercise 1: Linear Regression
    """

    # ==================== Part 0: Sigmoid funciton ====================

    print('Running warmUp exercise ... ')

    I = np.zeros((3, 1), dtype=float)
    I[0, 0], I[1, 0], I[2, 0] = 0.0, np.inf, -np.inf
    print(f'Matrix:\n{I}')
    print(sigmoid(I))

    print('Program paused. Press enter to continue.')
    input()

    # ==================== Part 1: Plotting ====================

    col_names = ['exam_1_score', 'exam_2_score', 'admission']
    data = pd.read_csv('ex2data1.txt', names=col_names, header=None, delimiter=',')

    x = data[['exam_1_score', 'exam_2_score']]
    y = data[['admission']]

    print('\nPlotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

    plotData1(data)

    print('\nProgram paused. Press enter to continue.\n')
    input()

    # ============ Part 2: Compute Cost and Gradient ============

    m, n = x.shape

    x['bias'] = [1 for _ in range(len(x))]
    x = x[['bias', 'exam_1_score', 'exam_2_score']]
    X, Y = x.to_numpy(), y.to_numpy()

    initial_theta = np.zeros((n + 1, 1))
    cost_null, grad_null = costFunction(initial_theta, X, y)
    print(f'Cost at initial theta:\n{n_decimal_places(cost_null, 3)}')
    print('Expected cost (approx): 0.693\n')
    print(f'Gradient at initial theta: {n_decimal_places(grad_null[0, 0], 4)} {n_decimal_places(grad_null[1, 0], 4)} {n_decimal_places(grad_null[2, 0], 4)}')
    print('Expected gradients (approx): -0.10000 -12.0092 -11.2628\n')
    test_theta = np.zeros((n + 1, 1), dtype=float)
    test_theta[0, 0], test_theta[1, 0], test_theta[2, 0] = -24, 0.2, 0.2
    cost_test, grad_test = costFunction(test_theta, X, y)
    print(f'Gradient at test theta: {n_decimal_places(grad_test[0, 0], 4)} {n_decimal_places(grad_test[1, 0], 4)} {n_decimal_places(grad_test[2, 0], 4)}')
    print('Expected gradients (approx): 0.043 2.566 2.647')
    print('\nProgram paused. Press enter to continue.\n')
    input()

    # ==================== Part 3: Optimizing ====================

    result = opt.minimize(f1, x0=np.array([0, 0, 0]), args=(X, Y))
    print(result)
    optimized_theta = result.x
    print('Expected cost (approx): 0.203\n')
    print('But, we found a better loss value.\n')
    print(f'Cost at theta found by scipy.optimize: {f1(optimized_theta, X, Y)}')


if __name__ == '__main__':
    ex2()


