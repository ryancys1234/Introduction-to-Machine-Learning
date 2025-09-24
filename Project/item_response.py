from utils import *
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta):
    # log_lklihood = 0
    # for r in range(len(data["user_id"])):
    #     i = data["user_id"][r]
    #     j = data["question_id"][r]
    #     cij = data["is_correct"][r]
    #     sig = sigmoid(theta[i]-beta[j])     
    #     log_lklihood += cij*np.log(sig) + (1-cij)*np.log(1-sig)

    t = np.array([ theta[i] for i in data['user_id'] ])
    b = np.array([ beta[j] for j in data['question_id'] ])
    c = np.array(data["is_correct"])
    sig = sigmoid(t-b)
    log_lklihood = np.sum(c*np.log(sig) + (1-c)*np.log(1-sig))
    return -log_lklihood

def update_theta_beta(data, lr, theta, beta):    
    d_theta = np.zeros(theta.shape[0]); d_beta = np.zeros(beta.shape[0])
    for r in range(len(data["user_id"])):
        i = data["user_id"][r]
        j = data["question_id"][r]
        cij = data["is_correct"][r]
        d_theta[i] += cij - sigmoid(theta[i] - beta[j])
        d_beta[j] += -cij + sigmoid(theta[i] - beta[j])
    theta += lr*d_theta; beta += lr*d_beta
    return theta, beta

def irt(data, val_data, lr, iterations):
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
        
        # print("Train NLLK: {} \t Train Score: {}".format(train_neg_lld, train_score))
        # print("Valid NLLK: {} \t Valid Score: {}".format(val_neg_lld, val_score))
        
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst

def evaluate(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.01; n_i = 12
    
    theta, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst = irt(train_data, val_data, lr, n_i)
    x = [i for i in range(n_i)]
    
    # plt.plot(x, train_acc_lst, label="Training"); plt.plot(x, val_acc_lst, label="Validation")
    # plt.title("Accuracy vs iteration"); plt.xlabel("Iteration"); plt.ylabel("Accuracy")
    # plt.legend()
    
    plt.plot(x, train_neg_lld_lst); plt.xlabel("Number of iterations"); plt.ylabel("Negative log-likelihood")
    plt.title("Training negative log-likelihood vs number of iterations")
    
    plt.plot(x, val_neg_lld_lst); plt.xlabel("Number of iterations"); plt.ylabel("Negative log-likelihood")
    plt.title("Validation log-likelihood vs number of iterations")
    
    print(f"Final validation accuracy: {evaluate(val_data, theta, beta)}")
    print(f"Final testing accuracy: {evaluate(test_data, theta, beta)}")
    
    questions = [100, 500, 1000]; theta_sorted = np.sort(theta)
    for q in questions:
        plt.plot(theta_sorted, sigmoid(theta_sorted - beta[q]), label=f'Question {q}')
    plt.xlabel("Theta"); plt.ylabel("Probability of correct response P(c_ij = 1)")
    plt.title("Probability of correct response vs theta")
    plt.legend()

if __name__ == "__main__":
    main()
