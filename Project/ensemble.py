from item_response import *
import numpy as np
np.random.seed(311)

def create_bootstrap(data):
    lst = np.random.randint(0, len(data['is_correct']), size=len(data['is_correct']))
    dct = {"user_id": [], "question_id": [], "is_correct": []}
    for i in lst:
        dct["user_id"].append(data["user_id"][i])
        dct["question_id"].append(data["question_id"][i])
        dct["is_correct"].append(data["is_correct"][i])
    return dct

def evaluate_ensemble(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(int(p_a >= 0.5))
    return pred

def average_pred(data, pred_lst):
    avg_pred = []
    for i in range(len(data['is_correct'])):
        avg_pred.append(((pred_lst[0][i] + pred_lst[1][i] + pred_lst[2][i]) / 3) >= 0.5)
    return np.sum((data["is_correct"] == np.array(avg_pred))) / len(data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    
    lr = 0.01; n_i = 12
    
    pred_lst_val = []
    for i in range(3):
        bootstrap_data = create_bootstrap(train_data)
        theta, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst = irt(bootstrap_data, val_data, lr, n_i)
        pred_lst_val.append(evaluate_ensemble(val_data, theta, beta))
    
    print(f"Final validation accuracy: {average_pred(val_data, pred_lst_val)}")
    
    pred_lst_test = []
    for i in range(3):
        bootstrap_data = create_bootstrap(train_data)
        theta, beta, train_acc_lst, val_acc_lst, train_neg_lld_lst, val_neg_lld_lst = irt(bootstrap_data, test_data, lr, n_i)
        pred_lst_test.append(evaluate_ensemble(test_data, theta, beta))
    
    print(f"Final test accuracy: {average_pred(test_data, pred_lst_test)} ")

if __name__ == "__main__":
    main()
