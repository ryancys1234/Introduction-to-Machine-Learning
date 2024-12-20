from utils import sigmoid
import numpy as np

def logistic_predict(weights, data):
    N = len(data)
    X = np.c_[data, np.ones(N)] # Adds the dummy feature to the data for bias
    z = np.dot(X, weights)
    y = sigmoid(z)
    return y

def evaluate(targets, y):
    N = len(targets)
    sum = 0
    for i in range(N): # Note: (targets[i]-1) and -(1-targets[i]) doesn't work due to an encountered uint8 bug
        t = targets[i]
        y_i = y[i]
        sum -= t*np.log(y_i)
        sum -= (1-t)*np.log(1-y_i)
    
    ce = float(sum / N)
    frac_correct = np.mean([int( targets[i] == int(y[i] >= 0.5) ) for i in range(N)]) # Computes the fraction using np.mean()
    
    # Debugging code:
    # train_targets = targets; i = np.random.randint(N-19, N)
    # print(f"Sample CE: -{train_targets[i]}*log({y[i]})-(1-{train_targets[i]})*log(1-{y[i]})")
    # print(f"Next 1: -{train_targets[i]}*{np.log(y[i])}-(1-{train_targets[i]})*{np.log(1-y[i])}")
    # print(f"Next 2: -{train_targets[i]*np.log(y[i])}-{(1-train_targets[i])*np.log(1-y[i])}")
    # print(f"Calculated: {float(-train_targets[i] * np.log( y[i] ) - (1-train_targets[i]) * np.log( 1 - y[i] ))}")
    
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    y = logistic_predict(weights, data)
    N = len(targets)
    X = np.c_[data, np.ones(N)] # Adds the dummy feature to the data for bias
    f = evaluate(targets, y)[0] # The averaged cross entropy loss
    df = (1/N)*X.T.dot([y_i - t_i for y_i, t_i in zip(y, targets)])
    return (f, df, y)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
