from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

# Note to TA: load_boston is deprecated
# from sklearn.datasets import load_boston

np.random.seed(311)

# +
# boston = load_boston() # Note to TA: this is deprecated
# x = boston['data']
# N = x.shape[0]
# x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
# d = x.shape[1]
# y = boston['target']

# Import and load boston housing prices dataset (could not directly load from sklearn.datasets)
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

x = np.concatenate((np.ones((506,1)),data),axis=1) #add constant one feature - no bias needed
N = x.shape[0]
d = x.shape[1]
y = target
# -

idx = np.random.permutation(range(N))


def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    # Preliminaries
    N_train = len(x_train)
    test_datum = np.reshape(test_datum, (d, 1)) # Reshapes the test vector into a matrix
    dist = l2(test_datum.T, x_train).flatten() # The L2 distance
    
    # Calculate the a's
    a_top = np.exp(-dist/(2*tau**2)) # Numerator of the a's
    a_bottom = np.sum(np.exp(-dist / (2*tau**2))) # Denominator of the a's
    a = [a_top[i]/a_bottom for i in range(N_train)] # List of the a's

    # Solve for w_star
    A = np.diag(a)
    lamI = lam*np.identity(d)
    w_star = np.linalg.solve((np.dot(x_train.T, np.dot(A, x_train)) + lamI), np.dot(x_train.T, np.dot(A, y_train)))
    
    return np.dot(test_datum.T, w_star)


def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    Output:
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    
    # Creating the 70% training and 30% validation sets using idx from above
    training_count = int(np.round(0.7*N))
    x_train, x_valid = x[idx[:training_count],:], x[idx[training_count:],:]
    y_train, y_valid = y[idx[:training_count],], y[idx[training_count:],]
    N_train = len(x_train)
    N_valid = len(x_valid)
    
    train_losses = []; valid_losses = [] # Loss vectors to be returned
    
    for tau in taus:
        train_loss = 0; valid_loss = 0 # Sums of losses for a given tau
        
        for i in range(N_train):
            y_hat = LRLS(x_train[i], x_train, y_train, tau, lam=1e-5) # Hard coded lambda value
            train_loss += 1/2*((y_hat-y_train[i])**2) # Adds the training loss
        
        for i in range(N_valid):
            y_hat = LRLS(x_valid[i], x_train, y_train, tau, lam=1e-5) # Hard coded lambda value
            valid_loss += 1/2*((y_hat-y_valid[i])**2) # Adds the validation loss
        
        train_losses.append(float(train_loss/N_train)) # Appends the averaged training loss
        valid_losses.append(float(valid_loss/N_valid)) # Appends the averaged validation loss
    
    return train_losses, valid_losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)

    # Plot for 4 (c)
    plt.semilogx(taus, train_losses, label="Training loss")
    plt.semilogx(taus, test_losses, label="Validation loss")
    plt.legend()
    plt.title("Tau vs average loss for locally weighted regression")
    plt.xlabel("Tau"); plt.ylabel("Average loss")
