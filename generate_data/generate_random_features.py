import numpy as np

def relu(x):
    """
    This method return the ReLU function.
    :param x: point (np.ndarray)
    :return:
    """
    return np.maximum(x, 0)

def phi(theta, X):
    """
    This method constructs random features
    :param theta: random features (np.ndarray)
    :param X: input data (np.ndarray)
    :return:
    """
    return relu(theta @ X.T).T

def return_RF(X, theta, dim):
    """
    This method returns random features in a certain dimension.
    :param X: input data (np.ndarray)
    :param theta: random features (np.ndarray)
    :param dim: chosen dimension of the random features
    :return: random features
    """
    assert theta.shape[1] == X.shape[1], 'This is not the correct dimension for theta and X '
    print('random feature dimension', dim)
    Phi = phi(theta[:dim, :], X)
    return Phi