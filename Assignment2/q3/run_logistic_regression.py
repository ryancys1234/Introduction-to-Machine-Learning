from check_grad import check_grad
from logistic import *
from utils import *
import matplotlib.pyplot as plt, numpy as np

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    N, M = train_inputs.shape
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 1000
    }
    weights = np.zeros((M + 1, 1))
    train_ce = []; valid_ce = []
    
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        train_ce.append(f) # Adds the cross entropy to training_ce for plotting
        weights -= hyperparameters["learning_rate"]*df
        
        f1, df1, y1 = logistic(weights, valid_inputs, valid_targets, hyperparameters)
        valid_ce.append(f1)
        
        if t == hyperparameters["num_iterations"] - 1:
            print(f"Training cross entropy loss: {f}, training classification accuracy: {evaluate(train_targets, y)[1]}")
            print(f"Validation cross entropy loss: {f1}, validation classification accuracy: {evaluate(valid_targets, y1)[1]}")
    
    test_inputs, test_targets = load_test()
    f2, df2, y2 = logistic(weights, test_inputs, test_targets, hyperparameters)
    print(f"Test cross entropy loss: {f2}, test classification accuracy: {evaluate(test_targets, y2)[1]}")
    
    t_values = [t for t in range(hyperparameters["num_iterations"])] # Used as the x-axis of the plots
    plt.plot(t_values, train_ce, label = "Training set")
    plt.plot(t_values, valid_ce, label = "Validation set")
    plt.xlabel("Iteration number"); plt.ylabel("Cross-entropy loss")
    plt.title("Averaged cross entropy loss vs iteration number")
    plt.legend()

if __name__ == "__main__":
    run_logistic_regression()
