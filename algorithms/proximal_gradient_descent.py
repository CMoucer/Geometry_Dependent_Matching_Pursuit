import numpy as np

def soft_thresh(x, l):
    """
    :param x: (np.ndarray)
    :param l: soft-thresholding paramater
    :return: soft-thresholding oprator
    """
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

## LASSO

def function_LASSO(X, y, w, lam):
    """
    This function evaluate the function value of the LASSO
    :param X: Data features
    :param y: Data to predict
    :param w: Point to evaluate
    :param lam: regularization parameter
    :return: function value for the LASSO
    """
    n = X.shape[0]
    return .5 / n * np.linalg.norm(X @ w - y)**2 + lam * np.linalg.norm(w, 1)

def proximal_gradient_descent_lasso(X, y, w_0, gamma, lam=.01, num_iters=100, mu=0.):
    """
    :param X: Data features
    :param y: Data to predict
    :param w_0: starting point
    :param gamma: step size
    :param lam: regularization parameter
    :param num_iters: number of iterations
    :return: iterates produced by the proximal gradient
    """

    #Initialisation of useful values
    ws = np.zeros((num_iters, w_0.shape[0]))
    ws[0] = w_0
    n = X.shape[0]
    #Looping until max number of iterations
    for i in range(1, num_iters):
        grad = X.T @ (X @ ws[i - 1] - y) / n + mu * ws[i - 1]
        ws[i] = soft_thresh(ws[i - 1] - gamma * grad, lam * gamma)
    return ws








