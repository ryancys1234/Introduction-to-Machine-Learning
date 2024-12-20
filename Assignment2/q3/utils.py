import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_train():
    with open('data/mnist_train.npz', 'rb') as f:
        train_set = np.load(f)
        train_inputs = train_set['train_inputs']
        train_targets = train_set['train_targets']
    return train_inputs, train_targets

def load_train_small():
    with open('data/mnist_train_small.npz', 'rb') as f:
        train_set_small = np.load(f)
        train_inputs_small = train_set_small['train_inputs_small']
        train_targets_small = train_set_small['train_targets_small']
    return train_inputs_small, train_targets_small

def load_valid():
    with open('data/mnist_valid.npz', 'rb') as f:
        valid_set = np.load(f)
        valid_inputs = valid_set['valid_inputs']
        valid_targets = valid_set['valid_targets']
    return valid_inputs, valid_targets

def load_test():
    with open('data/mnist_test.npz', 'rb') as f:
        test_set = np.load(f)
        test_inputs = test_set['test_inputs']
        test_targets = test_set['test_targets']
    return test_inputs, test_targets
