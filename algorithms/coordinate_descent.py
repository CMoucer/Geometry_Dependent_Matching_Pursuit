import numpy as np
from algorithms.proximal_gradient_descent import soft_thresh
from generate_data.sparse_optimum import generate_sparse_optimum

# LASSO REGRESSSION

def randomized_cd_lasso(X, y, w_0, lam=1, num_iters=10, lipschitz=True):
    """
    The method returns the iterates of randomized proximal coordinate descent
    :param X: input data (np.ndarray)
    :param y: label to predict (np.ndarray)
    :param w_0: starting point (np.ndarray)
    :param lam: regularization parameter
    :param num_iters: iteration number
    :param lipschiz: use the Lipschitz rule (apply the lipschitz parameter along coordinates)
    :return: iterates produced by randomized proximal coordinate decent
    """
    # Initialisation of useful values
    n, d = X.shape
    ws = np.zeros((num_iters, d))
    ws[0] = w_0
    Ls = np.diagonal((X.T @ X) / n)
    L = max(np.linalg.eigvalsh(X.T @ X)) / n
    # Looping until max number of iterations
    for i in range(1, num_iters):
        # give a new starting point
        ws[i] = ws[i-1].copy()
        k = np.random.randint(d)
        grad = X[:, k].T @ (X @ ws[i - 1] - y) / n
        if lipschitz:
            ws[i][k] = ws[i - 1][k] - 1/Ls[k] * grad
            ws[i][k] = soft_thresh(ws[i][k], lam / Ls[k])
        else:
            ws[i][k] = ws[i - 1][k] - 1/L * grad
            ws[i][k] = soft_thresh(ws[i][k], lam / L)
    return ws

def gsq_cd_lasso(X, y, w_0, lam=1, num_iters=10, lipschitz=True, L=0, Ls=0):
    """
    This method performs proximal coordinate descent with Gauss-Southwell rule.
    :param X: input data (np.ndarray)
    :param y: label to predict (np.ndarray)
    :param w_0: starting point (np.ndarray)
    :param lam: regularization parameter
    :param num_iters: iteration number
    :param lipschiz: Lipschitz rule or not
    :param L: Lipschitz parameter
    :param Ls: Local Lipschitz parameters (np.ndarray)
    :return:
    """
    # Initialisation of useful values
    n, d = X.shape
    ws = np.zeros((num_iters, d))
    ws[0] = w_0
    # Looping until max number of iterations
    for i in range(1, num_iters):
        # give a new starting point
        ws[i] = ws[i-1].copy()
        grad = X.T @ (X @ ws[i - 1] - y) / n
        if lipschitz:
            if (Ls == 0).any():
                Ls = np.diagonal((X.T @ X) / n)
            w = ws[i-1] - 1/Ls * grad
            for j in range(d):
                w[j] = soft_thresh(w[j], lam/Ls[j])
            progress = (w - ws[i - 1]) * grad + lam * np.abs(w) - lam * np.abs(ws[i - 1]) + .5 * Ls * (
                        w - ws[i - 1]) ** 2
        else:
            if L == 0:
                L = max(np.linalg.eigvalsh(X.T @ X)) / n
            w = ws[i - 1] - grad / L
            for j in range(d):
                w[j] = soft_thresh(w[j], lam/L)
            progress = (w - ws[i - 1]) * grad + lam * np.abs(w) - lam * np.abs(ws[i - 1]) + .5 * L * (
                        w - ws[i - 1]) ** 2
        k = np.argmin(progress)
        ws[i][k] = w[k]
    return ws


if __name__ == '__main__':
    d = 100
    n = 20
    s = 4
    sigma = 0.
    num_iters = 15
    print('optimal lambda in statistics', np.sqrt(np.log(d) / n))
    lam = 0.1

    w_0 = np.zeros(d)
    w_star = generate_sparse_optimum(d=d, s=s)
    print(w_star)


    X = np.random.multivariate_normal(np.zeros(d), np.eye(d), n)
    y = X @ w_star + np.random.normal(0, sigma, n)

    ws = gsq_cd_lasso(X=X, y=y, w_0=w_0, lam=lam, num_iters=num_iters)
    print(ws[-2:])

