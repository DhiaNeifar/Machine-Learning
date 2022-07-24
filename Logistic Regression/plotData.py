import matplotlib.pyplot as plt
import pandas as pd


def plotData1(data) -> None:
    pos = data[data['admission'] == 1]
    neg = data[data['admission'] == 0]

    fig = plt.figure()
    plt.plot(pos['exam_1_score'], pos['exam_2_score'], 'b+', label='Admitted')
    plt.plot(neg['exam_1_score'], neg['exam_2_score'], 'yo', label='Not Admitted')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.legend()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)
