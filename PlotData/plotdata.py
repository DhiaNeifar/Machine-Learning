import matplotlib.pyplot as plt


def plot_data(X, y):

    fig = plt.figure()
    plt.scatter(X, y)
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)
