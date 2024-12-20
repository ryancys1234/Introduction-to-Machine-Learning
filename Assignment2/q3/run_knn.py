from l2_distance import l2_distance
from utils import *
import matplotlib.pyplot as plt, numpy as np

def knn(k, train_data, train_labels, valid_data):
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]
    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)
    return valid_labels

def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    
    k_values = [1,3,5,7,9]; accuracies = []
    for k in k_values:
        count = 0
        pred_labels = knn(k, train_inputs, train_targets, valid_inputs)
        for i in range(len(pred_labels)):
            if pred_labels[i] == valid_targets[i]: count += 1
        accuracies.append(count / len(pred_labels))
    
    plt.title("Classification accuracy on validation set vs value of k in kNN on MNIST")
    plt.xlabel("Value of k"); plt.ylabel("Classification accuracy on validation set")
    plt.scatter(k_values, accuracies)
    plt.show()
    
    k_stars = [5,7,9]
    for k in k_stars:
        test_count = 0
        pred_test_labels = knn(k, train_inputs, train_targets, test_inputs)
        for i in range(len(pred_test_labels)):
            if pred_test_labels[i] == test_targets[i]: test_count += 1
        print(f"For k = {k}, the validation accuracy is {accuracies[k_values.index(k)]} and the test accuracy is {test_count / len(pred_test_labels)}")

if __name__ == "__main__":
    run_knn()
