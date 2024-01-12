import numpy as np

from algorithms.proximal_gradient_descent import soft_thresh


# Utils for the Chambolle-Pock algorithm
def evaluate_chambolle_pock_lagrangian(X, grad, L, lam, alpha, beta, eta, a):
    """

    :param X: data
    :param grad: data (that is here a gradient)
    :param L: smoothness parameter
    :param lam: regularization parameter
    :param alpha: Point at which the function is evaluated
    :param beta: Point at which the function is evaluated
    :param eta: Point at which the function is evaluated
    :param a: Point at which the function is evaluated
    :return:
    """
    return (X.T @ a) @ (beta - eta) + lam * np.linalg.norm(eta, 1) + L/2 * np.linalg.norm(beta - alpha, 1)**2 + X.T @ grad @ (beta - alpha)

def evaluate_chambolle_pock(X, grad, L, lam, alpha, beta, eta):
    """
    This function evaluates the sequence produced by CP in alpha, beta, eta.
    :param X: data
    :param grad: data (that is here a gradient)
    :param L: smoothness parameter (if known)
    :param lam: regularization parameter
    :param alpha: Point at which the function is evaluated
    :param beta: Point at which the function is evaluated
    :param eta: Point at which the function is evaluated
    :return:
    """
    return lam * np.linalg.norm(eta, 1) + L/2 * np.linalg.norm(beta - alpha, 1)**2 + X.T @ grad @ (beta - alpha)

def prox_l1_square(nu, tau):
    """
    :param nu: np.ndarray (dimension d)
    :param gamma: regularization of the squared l1-norm
    :param tau: regularization of the prox
    :return:
    """
    d = nu.shape[0]
    # Compute lambda
    sorted_nu = np.sort(np.abs(nu))[::-1]
    S = np.zeros(d)
    S[0] = sorted_nu[0] * np.sqrt(tau) / (1 + tau)
    for k in range(1, d):
        S[k] = S[k - 1] * (k * tau + 1) / ((k + 1) * tau + 1) + sorted_nu[k] / ((k + 1) * tau + 1) * np.sqrt(tau)
    lam = np.max(S) ** 2 / 2
    # Compute the solution to the problem
    eta = np.zeros(d) # eta trick
    beta = np.zeros(d) # solution to the problem
    for i in range(d):
        if lam > nu[i] ** 2 / tau / 2:
            beta[i] = 0
        else:
            eta[i] = np.abs(nu[i]) * np.sqrt(tau / lam / 2) - tau
            beta[i] = nu[i] / (1 + tau / eta[i])
    return beta, eta, 1 / 2 * sum([nu[i] ** 2 / (1 + eta[i] / tau) for i in range(d)])

def chambolle_pock_G(beta_0, eta_0, a_0, X, alpha, grad, tau, lam, sigma, theta=1, L=0, intermediate_iter=10):
    """
    :param beta_0: initialization np.ndarray of size d
    :param eta_0: initialization np.ndarray of size d
    :param a_0: initialization np.ndarray of size n
    :param X: data of size n x d
    :param alpha: starting point
    :param grad: gradient
    :param tau: parameter for chambolle pock such that tau x sigma x L_k **2 < 1
    :param lam: regularization parameter
    :param sigma: parameter for chambolle pock such that tau x sigma x L_k **2 < 1
    :param L: smoothness parameter of the function
    :param intermediate_iter: iteration number
    :return: iterates produced by CP
    """
    n, d = X.shape
    if L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    # Definition of the sequences
    a = np.zeros((intermediate_iter, n))
    beta = np.zeros((intermediate_iter, d))
    tilde_beta = np.zeros((intermediate_iter, d))
    eta = np.zeros((intermediate_iter, d))
    tilde_eta = np.zeros((intermediate_iter, d))
    # initialization
    a[0] = a_0
    beta[0], tilde_beta[0] = beta_0, beta_0
    eta[0], tilde_eta[0] = eta_0, eta_0
    for k in range(1, intermediate_iter):
        a[k] = a[k - 1] + sigma * X @ (tilde_beta[k - 1] - tilde_eta[k - 1])
        beta[k], _, _ = prox_l1_square(beta[k - 1] - tau * X.T @ (a[k] + grad) - alpha, tau * L)
        beta[k] += alpha
        # Prox on the l1 norm
        eta[k] = soft_thresh(eta[k - 1] + tau * X.T @ a[k], tau * lam)
        tilde_beta[k] = beta[k] + theta * (beta[k] - beta[k - 1])
        tilde_eta[k] = eta[k] + theta * (eta[k] - eta[k - 1])
    return beta, eta, a