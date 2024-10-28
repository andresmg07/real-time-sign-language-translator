# Model validation module. This module generates insights from the model performance results.

import pickle
import matplotlib.pyplot as plt
from os.path import join

import pandas as pd
import sklearn.svm
import numpy as np
import seaborn as sn

from classes.StatisticalClassifier import StatisticalClassifier
from util.plotter_util import plot_confusion_matrix, plot_bar_chart


def load_object(load_path):
    """
    Function that load the list of features from disk to python
    :param load_path: path in which the features are saved
    :return: list of tuples containing observations and labels
    """
    with open(join(load_path, "7500_features_dump.pkl"), "rb") as reader:
        return pickle.load(reader)


def load_model(load_path):
    """
    Function that load a pretrained model from disk
    :param load_path: path in which the model was saved
    :return: Classification model
    """
    with open(load_path, "rb") as reader:
        return pickle.load(reader)


def save_model(features, save_path):
    """
    Function that saves an array of features and labels to a pickle file
    :param save_path: path in which the features will be saved
    :param features: array of features to save
    """
    with open(join(save_path, "trained_model.pkl"), "wb") as writer:
        writer.write(pickle.dumps(features))


def show_confusion_matrix(confusion_matrix):
    sn.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.title("Matriz de confusión para la clasificación del abecedario ASL")
    plt.xlabel("Predicciones")
    plt.ylabel("Objetivo")
    plt.show()


def get_support_vector_machine_model(vectors):
    """
    Function that instances and trains the SVM model
    :param vectors: dictionary of labels and feature vectors
    :return: Trained SVM model
    """
    # Instance the model
    svm = sklearn.svm.SVC(kernel='linear', C=1.0)

    # Convert the data
    x = []
    y = []
    for _class in vectors.keys():
        x.extend(vectors[_class])
        y.extend([_class]*len(vectors[_class]))

    # Fit the model to the training data
    svm.fit(x, y)

    return svm

def get_classifier(trained_model_path=""):
    """
    Function that instantiates the classifier, trains it and returns it
    :param trained_model_path: if not empty, it will load a trained model
    :return: classifier and test dataset
    """
    if trained_model_path == "":
        statisticalClassifier = StatisticalClassifier()
        # Load default features
        vectors = load_object("./features/")
        # Split dataset into training and testing
        train_dataset, test_dataset = statisticalClassifier.split(vectors)
        # Train the classifier
        statisticalClassifier.fit(train_dataset)
        # return trained classifier and test dataset
        return statisticalClassifier, test_dataset
    else:
        return load_model(trained_model_path)


def main_statistical():
    """
    Procedure that retrieves features, trains the classifier and tests it
    """
    statisticalClassifier, test_dataset = get_classifier()
    score, confusion_matrix = statisticalClassifier.score(test_dataset)

    show_confusion_matrix(confusion_matrix)

    for i in range(confusion_matrix.shape[0]):
        print(f"Class {i}: accuracy: {confusion_matrix[i, i] / np.sum(confusion_matrix[i])}")

    print("Overall score: ", score)

    save_model(statisticalClassifier, "./model")

def main_SVM():
    statisticalClassifier = StatisticalClassifier()
    vectors = load_object("./features/")
    train, test = statisticalClassifier.split(vectors)
    svm = get_support_vector_machine_model(train)

    # Convert the test data
    x = []
    y = []
    for _class in test.keys():
        x.extend(test[_class])
        y.extend([_class]*len(test[_class]))

    prediction = svm.predict(x)

    classes_count = len(test.keys())
    confusion_matrix = np.zeros((classes_count, classes_count))
    test_keys = list(test.keys())
    test_keys.sort()
    class_map = {_class: index for index, _class in enumerate(test_keys)}

    for i in range(len(prediction)):
        confusion_matrix[class_map[y[i]], class_map[prediction[i]]] += 1

    letters = list(filter(lambda c: c != 'J' and c != 'Z', list(map(chr, range(65, 91)))))
    confusion_matrix_data_frame = pd.DataFrame(confusion_matrix, index=[i for i in letters], columns=[i for i in letters])

    plot_confusion_matrix(confusion_matrix_data_frame)

    accuracy_level_letter = []

    for i in range(confusion_matrix.shape[0]):
        current_accuracy = confusion_matrix[i, i] / np.sum(confusion_matrix[i])
        accuracy_level_letter.append(current_accuracy)
        print(f"Class {i}: accuracy: {current_accuracy}")

    print("Overall score: ", svm.score(x, y))

    plot_bar_chart(letters, accuracy_level_letter)




# Example validation run: python ./validator.py
if __name__ == '__main__':
    # This function will only be run if the current file is run directly
    main_SVM()

