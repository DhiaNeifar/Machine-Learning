import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y) -> None:

    fig = plt.figure()
    plt.plot(X, y, 'r+')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)


def plot_data_1(X, y, theta) -> None:
    fig = plt.figure()
    plt.plot(X[:, 1], y, 'r+')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.plot(X[:, 1], np.dot(X, theta))
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)
