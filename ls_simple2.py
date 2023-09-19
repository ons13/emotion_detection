"""Train emotion classifier using least squares."""

import numpy as np

def evaluate(A, Y, w):
    Yhat = np.argmax(A.dot(w), axis=1)
    return np.sum(Yhat == Y) / Y.shape[0]

def main():
    # load data
    with np.load('data/fer2013_train.npz') as data:
        X_train, Y_train = data['X'], data['Y']

    with np.load('data/fer2013_test.npz') as data:
        X_test, Y_test = data['X'], data['Y']

    # one-hot labels
    I = np.eye(6)
    Y_oh_train, Y_oh_test = I[Y_train], I[Y_test]
    d = 1000
    W = np.random.normal(size=(X_train.shape[1], d))
    # select first 100 dimensions
    A_train, A_test = X_train.dot(W), X_test.dot(W)

    # train model
    I = np.eye(A_train.shape[1])
    w = np.linalg.inv(A_train.T.dot(A_train) + 1e10 * I).dot(A_train.T.dot(Y_oh_train))

    # evaluate model
    print('(ridge) Train Accuracy:', evaluate(A_train, Y_train, w))
    print('(ridge) Test Accuracy:', evaluate(A_test, Y_test, w))

if __name__ == '__main__':
    main()