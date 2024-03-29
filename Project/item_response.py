from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


# +
def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

#     log_lklihood = 0
    
#     for r in range(len(data["user_id"])):
#         i = data["user_id"][r]
#         j = data["question_id"][r]
#         cij = data["is_correct"][r]
#         sig = sigmoid(theta[i]-beta[j])
        
#         log_lklihood += cij*np.log(sig) + (1-cij)*np.log(1-sig)

    t = np.array([ theta[i] for i in data['user_id'] ])
    b = np.array([ beta[j] for j in data['question_id'] ])
    c = np.array(data["is_correct"])
    sig = sigmoid(t-b)
    
    log_lklihood = np.sum(c*np.log(sig) + (1-c)*np.log(1-sig))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


# -

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    d_theta = np.zeros(theta.shape[0]); d_beta = np.zeros(beta.shape[0])
    
    for r in range(len(data["user_id"])):
        i = data["user_id"][r]
        j = data["question_id"][r]
        cij = data["is_correct"][r]
        
        d_theta[i] += cij - sigmoid(theta[i] - beta[j])
        d_beta[j] += -cij + sigmoid(theta[i] - beta[j])
    
    theta += lr*d_theta; beta += lr*d_beta
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


# +
def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542); beta = np.zeros(1774)

    train_acc_lst = []; val_acc_lst = []; train_neg_lld_lst = []; val_neg_lld_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        
        train_neg_lld_lst.append(train_neg_lld)
        val_neg_lld_lst.append(val_neg_lld)
        
        train_score = evaluate(data=data, theta=theta, beta=beta)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        
        train_acc_lst.append(train_score)
        val_acc_lst.append(val_score)
        
#         print("Train NLLK: {} \t Train Score: {}".format(train_neg_lld, train_score))
#         print("Valid NLLK: {} \t Valid Score: {}".format(val_neg_lld, val_score))
        
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst


# -

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


# +
def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    
    lr = 0.01; n_i = 12
    
    theta, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst = irt(train_data, val_data, lr, n_i)
    x = [i for i in range(n_i)]
    
# #     Accuracy plot
#     plt.plot(x, train_acc_lst, label="Training"); plt.plot(x, val_acc_lst, label="Validation")
#     plt.title("Accuracy vs iteration"); plt.xlabel("Iteration"); plt.ylabel("Accuracy")
#     plt.legend()
    
#     Training negative log-likelihood plot
    plt.plot(x, train_neg_lld_lst); plt.xlabel("Number of iterations"); plt.ylabel("Negative log-likelihood")
    plt.title("Training negative log-likelihood vs number of iterations")
    
#     Validation negative log-likelihood plot
    plt.plot(x, val_neg_lld_lst); plt.xlabel("Number of iterations"); plt.ylabel("Negative log-likelihood")
    plt.title("Validation log-likelihood vs number of iterations")
    
    print(f"Final validation accuracy: {evaluate(val_data, theta, beta)}")
    print(f"Final testing accuracy: {evaluate(test_data, theta, beta)}")
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    
    questions = [100, 500, 1000]; theta_sorted = np.sort(theta)
    for q in questions:
        plt.plot(theta_sorted, sigmoid(theta_sorted - beta[q]), label=f'Question {q}')
    
    plt.xlabel("Theta"); plt.ylabel("Probability of correct response P(c_ij = 1)")
    plt.title("Probability of correct response vs theta")
    plt.legend()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


# -

if __name__ == "__main__":
    main()
