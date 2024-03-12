# %load_ext autoreload
# %autoreload 2

from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
#     train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    
    hyperparameters = {
        "learning_rate": 0.1, # 0.1 for nmist_train, 0.01 for mnist_train_small
        "weight_regularization": 0.,
        "num_iterations": 1000 # 1000 for nmist_train, 200 for nmist_train_small
    }
    weights = np.zeros((M + 1, 1))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    
    train_ce = [] # For plotting the cross entropy for the training set
    valid_ce = [] # For plotting the cross entropy for the validation set
    
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        train_ce.append(f) # Adds the cross entropy to training_ce for plotting
        weights -= hyperparameters["learning_rate"]*df # Updates gradient descent rule
        
        f1, df1, y1 = logistic(weights, valid_inputs, valid_targets, hyperparameters)
        valid_ce.append(f1) # Adds the cross entropy to valid_ce for plotting
        
        if t == hyperparameters["num_iterations"] - 1: # Prints the final training loss metrics
            print(f"Training cross entropy loss: {f}, training classification accuracy: {evaluate(train_targets, y)[1]}")
            print(f"Validation cross entropy loss: {f1}, validation classification accuracy: {evaluate(valid_targets, y1)[1]}")
    
    # The testing set is only used after the optimal hyperparameters have been selected
    test_inputs, test_targets = load_test()
    f2, df2, y2 = logistic(weights, test_inputs, test_targets, hyperparameters)
    print(f"Test cross entropy loss: {f2}, test classification accuracy: {evaluate(test_targets, y2)[1]}")
    
    # Plots for 3.2 (c)
    t_values = [t for t in range(hyperparameters["num_iterations"])] # Used as the x-axis of the plots
    plt.plot(t_values, train_ce, label = "Training set")
    plt.plot(t_values, valid_ce, label = "Validation set")
    plt.xlabel("Iteration number"); plt.ylabel("Cross-entropy loss")
    plt.title("Averaged cross entropy loss vs iteration number")
    plt.legend()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10
    
    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)
    
    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)
    
    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
