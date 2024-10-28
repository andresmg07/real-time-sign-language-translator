###############################################################################################################
#                                                                                                             #
#   This file contains the class StatisticalClassifier which is used to classify the observations of          #
#   a multi class model                                                                                       #
#                                                                                                             #
###############################################################################################################

import math
import numpy as np


class StatisticalClassifier:
    def __init__(self):
        """
        Constructor that initializes the attributes of the class
        """
        self.average_vectors = {}
        self.variance_vectors = {}
        self.training_size = 0

    def compute_each_class_prob(self, vector):
        """
        Method that calculates the probability of a vector of being classified as either class
        :param vector: features vector
        :return: probabilities dict
        """
        vector = np.array(vector)
        len_vector = len(vector)
        prob_dict = {}
        for key in self.average_vectors:
            test_vector = ((vector - self.average_vectors[key]) ** 2)
            prob_dict[key] = sum(
                [1 if variance >= test else 0 for test, variance in
                 zip(test_vector, self.variance_vectors[key])]) / len_vector

        return prob_dict

    def split(self, dataset, test_ratio=0.25):
        """
        Method that splits the dataset into training and testing datasets
        :param dataset: dictionary of features vectors arranged by class
        :param test_ratio: percentage of data to be used for testing
        :return: train and test datasets
        """
        test_dict = {}
        train_dict = {}
        for _class in dataset.keys():
            dataset[_class] = np.array(dataset[_class])
            np.random.shuffle(dataset[_class])
            split_index = math.floor(dataset[_class].shape[0] * test_ratio)
            test_dict[_class] = dataset[_class][0:split_index]
            train_dict[_class] = dataset[_class][split_index:]

        return train_dict, test_dict

    def fit(self, train_dict):
        """
        Method that trains the classifier with a data set
        :param train_dict: data dict
        """
        item_count = 0
        for key in train_dict.keys():
            features_vectors = train_dict[key]
            item_count += features_vectors.shape[0]
            average_vector = features_vectors.sum(axis=0) / len(train_dict[key])
            self.average_vectors[key] = average_vector
            self.variance_vectors[key] = np.sum((features_vectors - average_vector) ** 2, axis=0) / \
                                         features_vectors.shape[0]

        self.training_size = item_count

    def predict(self, vectors):
        """
        Method that predicts the labels of the given vectors
        :param vectors:
        :return:
        """
        predictions = []
        for vector in vectors:
            if type(vector) is tuple:
                vector = vector[0]
            probs_dict = self.compute_each_class_prob(vector)
            predictions.append(max(probs_dict, key=probs_dict.get))

        return predictions

    def score(self, test_dict):
        """
        Method that computes the accuracy level of the model compared to a test dataset
        :param test_dict: test dict
        :return: accuracy level, confusion matrix
        """
        hits = 0
        test_total = 0
        classes_count = len(test_dict.keys())
        confusion_matrix = np.zeros((classes_count, classes_count))
        tested_count = 0
        n = min([len(x) for x in test_dict.values()])
        class_map = {_class: index for index, _class in enumerate(test_dict.keys())}
        for _class in test_dict.keys():
            class_array = test_dict[_class]
            tested_count += class_array.shape[0]
            for i, observation in enumerate(class_array):
                if i == n:
                    break
                probs_dict = self.compute_each_class_prob(observation)
                predicted_label = max(probs_dict, key=probs_dict.get)
                if predicted_label == _class:
                    hits += 1
                test_total += 1
                confusion_matrix[class_map[_class], class_map[predicted_label]] += 1

        return hits / test_total, confusion_matrix
