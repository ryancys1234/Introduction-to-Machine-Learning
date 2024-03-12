from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    
    N = len(data)
    X = np.c_[data, np.ones(N)] # Adds the dummy feature to the data for bias
    z = np.dot(X, weights)
    y = sigmoid(z)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


# +
def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    
    N = len(targets)
    sum = 0
    for i in range(N): # Note: (targets[i]-1) and -(1-targets[i]) doesn't work due to an encountered uint8 bug
        t = targets[i]
        y_i = y[i]
        sum -= t*np.log(y_i)
        sum -= (1-t)*np.log(1-y_i)
    
    ce = float(sum / N)
    frac_correct = np.mean([int( targets[i] == int(y[i] >= 0.5) ) for i in range(N)]) # Computes the fraction using np.mean()
    
#     # Debugging code:
#     train_targets = targets; i = np.random.randint(N-19, N)
#     print(f"Sample CE: -{train_targets[i]}*log({y[i]})-(1-{train_targets[i]})*log(1-{y[i]})")
#     print(f"Next 1: -{train_targets[i]}*{np.log(y[i])}-(1-{train_targets[i]})*{np.log(1-y[i])}")
#     print(f"Next 2: -{train_targets[i]*np.log(y[i])}-{(1-train_targets[i])*np.log(1-y[i])}")
#     print(f"Calculated: {float(-train_targets[i] * np.log( y[i] ) - (1-train_targets[i]) * np.log( 1 - y[i] ))}")
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


# -

def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of par ameters, and the probabilities given by   #
    # logistic regression.                                              #
    #####################################################################
    
    N = len(targets)
    X = np.c_[data, np.ones(N)] # Adds the dummy feature to the data for bias
    f = evaluate(targets, y)[0] # The averaged cross entropy loss
    df = (1/N)*X.T.dot([y_i - t_i for y_i, t_i in zip(y, targets)])
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
