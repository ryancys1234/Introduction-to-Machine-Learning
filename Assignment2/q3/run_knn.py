from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    
    # Code for part (d) 1. (a)
    k_values = [1,3,5,7,9]
    accuracies = [] # List of accuracies to be used for plotting
    
    for k in k_values:
        count = 0 # Counts the number of corrected predicted validation data points for this k
        pred_labels = knn(k, train_inputs, train_targets, valid_inputs)
        
        for i in range(len(pred_labels)):
            if pred_labels[i] == valid_targets[i]: count += 1 # Increments the count
        
        accuracies.append(count / len(pred_labels)) # Adds the validation accuracy rate to the list of accuracies
    
    plt.title("Classification accuracy on validation set vs value of k in kNN on MNIST")
    plt.xlabel("Value of k"); plt.ylabel("Classification accuracy on validation set")
    plt.scatter(k_values, accuracies)
    plt.show()
    
    # Code for part (d) 1. (b)
    k_stars = [5,7,9]
    
    for k in k_stars:
        test_count = 0 # Counts the number of corrected predicted test data points for this k
        pred_test_labels = knn(k, train_inputs, train_targets, test_inputs)
        
        for i in range(len(pred_test_labels)):
            if pred_test_labels[i] == test_targets[i]: test_count += 1 # Increments the count
        
        # Prints the validation accuracy rate and the test accuracy rate for this k
        print(f"For k = {k}, the validation accuracy is {accuracies[k_values.index(k)]} and the test accuracy is {test_count / len(pred_test_labels)}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
