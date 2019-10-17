import numpy as np
import matplotlib.pyplot as plt


def visualize_dual_probabilities(p1, p2, domain, title,ylabel="Probability",p1_labels=["User One"],p2_labels=["User Two"]):
    x = np.arange(1, domain, 1)



    # multiple line plot
    # plt.gca().set_ylim([0,1])
    for column in range(len(p1.T)):
        print(p1_labels[column])
        plt.plot(x,p1[:,column],label=p1_labels[column],linewidth=1)

    for column in range(len(p2.T)):
        plt.plot(x,p2[:,column],label=p2_labels[column],linewidth=1)
   

    plt.legend(loc="best")
    
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()