import numpy as np
import matplotlib.pyplot as plt


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


def plot_data_surface(x, y, z) -> None:

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, z, cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('J_value')
    plt.show()
    plt.close(fig)


def plot_data_contour(x, y, Z) -> None:

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(1, 1)
    lvl = np.logspace(-2, 3, 20)
    CS = plt.contour(X, Y, Z, levels=lvl)
    ax.clabel(CS, inline=True, fontsize=10)

    plt.show()
    plt.close(fig)


def plot_convergence(J) -> None:

    fig = plt.figure()
    plt.plot([i for i in range(len(J))], J[:len(J)], 'b')
    plt.xlabel('number of iterations')
    plt.ylabel('Cost function J')
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)


def plot_multi_convergence(J, clrs, alphas):
    for index, elm in enumerate(J):
        plt.plot([i for i in range(len(J[index]))], J[index][:len(J[index])], clrs[index], label=f'lr={alphas[index]}')
    plt.xlabel("number of iterations")
    plt.ylabel("Cost function")
    plt.title("J = f(lr)")
    plt.legend()
    plt.show()
