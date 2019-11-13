import numpy as np
import matplotlib.pyplot as plt


def visualize_probabilities(p1, p2, domain, title, ylabel="Probability", p1_labels=["User One"], p2_labels=["User Two"]):
    x = np.arange(1, domain, 1)
    # multiple line plot
    for column in range(len(p1.T)):
        plt.plot(x, p1[:, column], label=p1_labels[column], linewidth=1)
    for column in range(len(p2.T)):
        plt.plot(x, p2[:, column], label=p2_labels[column], linewidth=1)
    plt.legend(loc="best")
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def opposing_probabilities(p1,p2,title):
    """
        This is meant to plot two probabilities in relation to one another.
        x axis - probability of A happening for player one
        y axis - probability of A happening for player two

    """
    colors = (0,0,0)
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    area = 0.5
    plt.scatter(p1, p2, s=area, c=colors, alpha=0.5)
    plt.title(title)
    plt.xlabel('Player One')
    plt.ylabel('Player Two')
    plt.show()
