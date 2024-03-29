import numpy as np
from algorithms.proximal_gradient_descent import soft_thresh
from generate_data.gaussian import generate_uniform_gaussian_data
from generate_data.logreg_data import log_reg_function_reg, gradient_logreg, log_reg_function

## LASSO REGRESSION

def fista(X, y, w_0, lam, gamma, num_iters, mu=0.):
    """
    Returns the iterates produced by FISTA
    :param X: data feature (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param w_0: starting point (np.ndarray)
    :param lam: regularization parameter
    :param gamma: step size parameter
    :param num_iters: number of iterations
    :return:
    """
    n, d = X.shape[0], X.shape[1]
    ws = np.zeros((num_iters, d))
    xs = np.zeros((num_iters, d))
    ws[0], xs[0] = w_0, w_0
    t = 1
    for i in range(1, num_iters):
        grad = - 1/n * X.T.dot(y - X.dot(xs[i-1])) + mu * xs[i-1]
        ws[i] = soft_thresh(xs[i-1] - grad * gamma, lam * gamma)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        xs[i] = ws[i] + ((t0 - 1.) / t) * (ws[i] - ws[i-1])
    return ws

