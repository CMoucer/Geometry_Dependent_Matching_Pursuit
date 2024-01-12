import numpy as np
from generate_data.sparse_optimum import generate_sparse_optimum, generate_sparse_simplex

def log_reg_function(X, y, w, mu=0):
    n = X.shape[0]
    f = 0
    for i in range(n):
        f += 1 / n * np.log(1 + np.exp(-y[i] * np.dot(w, X[i])))
    # Add a possible strongly convex parameter
    f += mu / 2 * np.linalg.norm(w) ** 2
    return f

def log_reg_function_reg(X, y, w, lam=0, ord=1, mu=0):
    n = X.shape[0]
    f = 0
    for i in range(n):
        f += 1 / n * np.log(1 + np.exp(-y[i] * np.dot(w, X[i])))
    f += mu / 2 * np.linalg.norm(w, ord=2) ** 2
    f += lam * np.linalg.norm(w, ord=ord)
    return f

def gradient_logreg(X, y, w, mu=0):
    n, d = X.shape
    grad = np.zeros(d)
    for j in range(n):
        grad -= y[j] * X[j] / (1 + np.exp(y[j] * np.dot(w, X[j]))) / n
    grad += mu * w
    return grad

def generate_data_points_logistic_regression(d, n, sigma=1, nu=1, s=0):
    """
    :param d: dimension
    :param n: data points
    :param sigma: noise parameter
    :return:
    """

    ## Generate X as a normal distribution
    X = np.random.multivariate_normal(np.zeros(d), nu*np.eye(d), n)
    if s==0:
        # Plot a random vector
        w_ = 2 * np.random.rand(d) - 1
    else:
        w_ = generate_sparse_optimum(d=d, s=s)
    ## noise :
    E = np.random.normal(0, sigma, n)
    # Model for logistic regression
    p_noisy = 1 / (1 + np.exp(-w_ @ X.T + E))
    Y = 2 * np.random.binomial(1, p_noisy) - 1
    return Y, X, w_, E

def generate_data_points_logistic_regression_simplex(d, n, sigma=1, nu=1, s=0, R=1.2):
    """
    :param d: dimension
    :param n: data points
    :param sigma: noise parameter
    :return:
    """

    ## Generate X as a normal distribution
    X = np.random.multivariate_normal(np.zeros(d), nu*np.eye(d), n)
    if s==0:
        # Plot a random vector
        w_ = 2 * np.random.rand(d) - 1
    else:
        w_ = generate_sparse_simplex(d=d, s=s, R=R)
    ## noise :
    E = np.random.normal(0, sigma, n)
    # Model for logistic regression
    p_noisy = 1 / (1 + np.exp(-w_ @ X.T + E))
    Y = 2 * np.random.binomial(1, p_noisy) - 1
    return Y, X, w_, E
