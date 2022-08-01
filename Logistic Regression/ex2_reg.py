import numpy as np
import pandas as pd
import scipy.optimize as opt


from plotData import plotData_reg, plotDecisionBoundary_reg
from mapfeature import map_feature
from costFunction import cost_function_reg, cost_reg
from utils import n_decimal_places, n_decimal_numarray, display


def ex2_reg():

    """
        Machine Learning Online Class - Exercise 2: Logistic Regression With Regularization
    """

    # ==================== Part 0: Plotting ====================

    col_names = ['test_1', 'test_2', 'admission']
    data = pd.read_csv('ex2data2.txt', names=col_names, header=None, delimiter=',')

    x = data[['test_1', 'test_2']]
    y = data[['admission']]

    print('\nPlotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

    plotData_reg(data)

    print('\nProgram paused. Press enter to continue.\n')
    input()

    # =========== Part 1: Regularized Logistic Regression ============

    X = map_feature(x, 6)
    X, Y = X.to_numpy(), y.to_numpy()
    m, n = X.shape

    initial_theta = np.zeros((n, 1))
    lamda = 1

    initial_cost, initial_grad = cost_function_reg(initial_theta, X, Y, lamda)

    print(f'Cost at initial theta (zeros): {n_decimal_places(initial_cost, 3)}')
    print('Expected cost (approx): 0.693\n')
    print(f'Gradient at initial theta (zeros) - first five values only: {display(n_decimal_numarray(initial_grad[:5], 4))}')
    print('Expected gradients (approx) - first five values only: 0.0085 0.0188 0.0001 0.0503 0.0115')
    print('\nProgram paused. Press enter to continue.\n')
    input()

    test_theta = np.ones((n, 1))
    test_cost, test_grad = cost_function_reg(test_theta, X, y, 10)
    print(f'Cost at test theta (zeros): {n_decimal_places(test_cost, 2)}')
    print('Expected cost (approx): 3.16\n')
    print(f'Gradient at test theta (zeros) - first five values only: {display(n_decimal_numarray(test_grad[:5], 4))}')
    print('Expected gradients (approx) - first five values only: 0.3460 0.1614 0.1948 0.2269 0.0922')
    print('\nProgram paused. Press enter to continue.\n')
    input()

    # =========== Part 2: Regularization and Accuracies ============

    result = opt.minimize(cost_reg, x0=np.array([0 for _ in range(n)]), args=(X, Y, lamda), method='BFGS')
    optimized_theta = result.x
    print(f'Cost minimized {n_decimal_places(result.fun)}')

    plotDecisionBoundary_reg(data, optimized_theta)


if __name__ == '__main__':
    ex2_reg()
