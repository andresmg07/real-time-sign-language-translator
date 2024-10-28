# Utilitarian module used to plot the results from the classification model

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sn


def plot_confusion_matrix(confusion_matrix):
    """
    Procedure that plots the confusion matrix
    :param confusion_matrix: numpy ndarray that represents the confusion matrix
    """
    sn.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.xlabel("Predicción")
    plt.ylabel("Objetivo")
    plt.show()


def plot_bar_chart(labels, data):
    """
    Procedure that plots the given data as a bar chart
    :param labels: List of labels associated with the given data.
    :param data: Data matrix used to plot the bar chart
    """
    plt.bar(labels, data, color='blue')
    plt.xlabel("Letras")
    plt.ylabel("Efectividad de predicción")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.show()