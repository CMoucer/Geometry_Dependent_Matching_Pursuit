import numpy as np

from algorithms.proximal_gradient_descent import soft_thresh

def alternating_minimization(X, lam, alpha, Q2, grad, eta_0, gamma_0, L=0, intermediate_iters=10, epsilon=0):
    """
    This method optimize a function (defined in supplementary material) using an alternating minimization technique.
    :param X: input data (np.ndarray)
    :param lam: regularization parameter
    :param alpha: starting point (np.ndarray)
    :param Q2: Basis for Kernel of X (np.ndarray)
    :param grad: gradient at past iterate (np.ndarray)
    :param eta_0: starting point (np.ndarray)
    :param gamma_0: starting eta-trick parameter in the simplex
    :param L: smoothness with respect to l_1 norm
    :param num_iters: iteration number
    :param model: model 'lasso' or none or 'logreg'
    :param epsilon: bias for the alternating minimization procedure
    :return:
    """
    n, d = X.shape
    if L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    # Compute the basis for Ker(P)
    r = Q2.shape[1]
    # Store the values
    zs = np.zeros((intermediate_iters, r))
    etas = np.zeros((intermediate_iters, d))
    gammas = np.ones((intermediate_iters, d))/d
    # Initialization
    etas[0], gammas[0] = eta_0, gamma_0
    for i in range(1, intermediate_iters):
        # optimize over z
        zs[i] = np.linalg.inv(Q2.T @ np.diag(1 / gammas[i-1]) @ Q2) @ Q2.T @ np.diag(1/gammas[i-1]) @ (alpha - etas[i-1])
        # optimize over gamma
        u = np.sqrt((etas[i-1] + Q2 @ zs[i] - alpha)**2 + epsilon)
        gammas[i] = u / np.linalg.norm(u, ord=1)
        # optimize over eta
        etas[i] = soft_thresh(alpha - np.diag(gammas[i] / L) @ X.T @ grad - Q2 @ zs[i], lam / L * gammas[i])
    return etas, gammas, zs


def ar_bcd(X, lam, alpha, Q2, grad, eta_0, gamma_0, L=0, p_1=0.5, p_2=0.5,  intermediate_iters=10, epsilon=0):
    """
    This method optimize a function defined in Supplementary Material using the AR-BCD optimization method.
    :param X: Data (np.ndarray)
    :param lam: regularization parameter
    :param alpha: reference point (np.ndarray)
    :param eta_0: starting point (np.ndarray)
    :param gamma_0: starting point (np.ndarray)
    :param Q2: Basis for the kernel of X
    :param p_1: probability for first block
    :param p_2: probability for second block
    :param L: smoothness paramater (equal to L_1)
    :param intermediate_iters: iteration number of the inner loop
    :param model: 'lasso' or 'logreg'
    :param epsilon: regularization for convergence
    :return:
    """
    n, d = X.shape
    assert p_1 == 1 - p_2
    if L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    r = Q2.shape[1]
    # Store the values
    zs = np.zeros((intermediate_iters, r))
    etas = np.zeros((intermediate_iters, d))
    gammas = np.ones((intermediate_iters, d))/d
    thetas = np.ones((intermediate_iters, d))/d
    ks = np.zeros(intermediate_iters)
    # Initialization
    etas[0], gammas[0] = eta_0, gamma_0
    for i in range(1, intermediate_iters):
        ks[i] = np.random.binomial(1, p_1)
        if ks[i] == 0.:
            zs[i] = np.linalg.inv(Q2.T @ np.diag(1 / gammas[i-1]) @ Q2) @ Q2.T @ np.diag(1/gammas[i-1]) @ (alpha - etas[i-1])
            etas[i] = etas[i - 1]
        elif ks[i] == 1.:
            etas[i] = np.diag(thetas[i-1] * gammas[i-1] / (L * thetas[i-1] + lam * gammas[i-1])) @ ((L * np.diag(1 / gammas[i-1])) @ (alpha - Q2 @ zs[i-1]) - X.T @ grad)
            zs[i] = zs[i - 1]
        # optimize over gamma
        u = np.sqrt((etas[i] + Q2 @ zs[i] - alpha)**2 + epsilon)
        gammas[i] = u / np.linalg.norm(u, ord=1)
        thetas[i] = np.sqrt(etas[i] ** 2 + epsilon)
    return etas, gammas, zs, thetas


def evaluate_function_am(X, y, eta, z, gamma, Q2, alpha, grad, lam, L=0, model='lasso'):
    n, d = X.shape
    if model == 'lasso':
        grad = 1 / n * (X @ alpha - y)
    elif L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    func = grad.T @ (X @ (eta - alpha)) + lam * np.linalg.norm(eta, ord=1) + L/2 * (eta + Q2 @ z - alpha).T @ np.diag(1/gamma) @ (eta + Q2 @ z - alpha)
    return func


def evaluate_function_arbcd(X, y, alpha, Q2, lam, eta, z, gamma, theta, L=0, model='lasso'):
    """
    This method evaluate a well-chosen function that is optimized with AR-BCD.
    """
    n, d = X.shape
    if model == 'lasso':
        grad = 1 / n * (X @ alpha - y)
    elif L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    func = 0
    func += grad.T @ (X @ (eta - alpha)) + L / 2 * (eta + Q2 @ z - alpha).T @ np.diag(1 / gamma) @ (
                eta + Q2 @ z - alpha)
    func += lam / 2 * (eta.T @ np.diag(1 / theta) @ eta + np.sum(theta))
    return func


def evaluate_original_function_am(X, y, eta, beta, alpha, lam, L=0, model='lasso'):
    """
        This method evaluate a well-chosen function that is optimized with AR-BCD.
    """
    n, d = X.shape
    if model == 'lasso':
        grad = 1 / n * (X @ alpha - y)
    elif L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    func = 0
    func += grad.T @ (X @ (beta - alpha)) + L/2 * np.linalg.norm(beta - alpha, 1) ** 2 + lam * np.linalg.norm(eta, 1)
    return func