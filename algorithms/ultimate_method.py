import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from generate_data.logreg_data import gradient_logreg
from algorithms.proximal_gradient_descent import proximal_gradient_descent_lasso
from algorithms.chambolle_pock import chambolle_pock_G
from algorithms.alternating_minimization import alternating_minimization, ar_bcd


## Ultimate matching pursuit for the LASSO

def ultimate_method(X, y, w_0, lam=1, num_iters=10):
    """
    This method returns the iterates produces by the ultimate matching pursuit using CVXPY and the solver MOSEK.
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param w_0: starting point (np.ndarray)
    :param lam: regularization parameter
    :param num_iters: iteration number
    :return: sequence produced by the method
    """
    n, d = X.shape
    ws = np.zeros((num_iters, n))
    betas = np.zeros((num_iters, d))
    betas_nonsparse = np.zeros((num_iters, d))
    ws[0] = w_0
    # Compute the smoothness parameter
    L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    # Compute the oracle and z_min
    for i in range(1, num_iters):
        # compute the minimal value for z with an LMO
        grad = 1 / n * (ws[i - 1] - y)
        # Perform a verification with cvxpy
        eta = cp.Variable(d)
        beta = cp.Variable(d)
        constraints = [X @ eta == X @ (beta - betas[i - 1]) + ws[i - 1]]
        objective = cp.Minimize(grad.T @ (X @ (beta - betas[i - 1])) + L / 2 * cp.norm(beta - betas[i - 1], 1) ** (2) + lam * cp.norm(eta, 1))
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_BASIS_TOL_X': 1.0e-7})
        ws[i] = X @ beta.value
        betas[i] = eta.value
        betas_nonsparse[i] = beta.value

    return betas, betas_nonsparse

## Ultimate matching pursuit with inner loop strategies for the LASSO

def ultimate_chambolle_pock(X, y, beta_0, lam=1, sigma=1, tau=1, theta=1, num_iters=10, intermediate_iter=np.ones(10)):
    """
    This method returns the ultimate matching pursuit with an inner loop solved using the Chambolle-Pock algorithm.
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param beta_0: starting point (np.ndarray)
    :param lam: regularization parameter
    :param sigma: CP parameter
    :param tau: CP parameter
    :param theta: CP parameter
    :param num_iters: iteration number of the outer loop
    :param intermediate_iter: iteration number of the inner loop
    :return: sequences produced by the method
    """
    n, d = X.shape
    ws = np.zeros((num_iters, n))
    betas = np.zeros((num_iters, d))
    etas = np.zeros((num_iters, d))
    mean_betas = np.zeros((num_iters, d))
    mean_etas = np.zeros((num_iters, d))
    a_s = np.zeros((num_iters, n))
    # Initialization
    betas[0], etas[0] = beta_0, beta_0
    ws[0] = X @ beta_0
    L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    assert len(intermediate_iter) == num_iters, 'the number of intermediate_iter is not equal to the num_iters'
    # Compute the oracle and z_min
    a_0 = np.zeros(n)
    for i in range(1, num_iters):
        # compute the minimal value for z with an LMO
        grad = 1 / n * (ws[i - 1] - y)
        # CHAMBOLLE POCK FOR AN INTERMEDIATE ITERATION NUMBER
        beta, eta, a = chambolle_pock_G(beta_0=betas[i - 1],
                                    eta_0=etas[i - 1],
                                    a_0=a_0,
                                    X=X,
                                    alpha=betas[i - 1],
                                    grad=grad,
                                    tau=tau,
                                    lam=lam,
                                    sigma=sigma,
                                    theta=theta,
                                    L=L,
                                    intermediate_iter=intermediate_iter[i])
        betas[i] = beta[-1]
        etas[i] = eta[-1]
        mean_betas[i] = np.mean(beta, axis=0)
        mean_etas[i] = np.mean(eta, axis=0)
        a_0 = a[-1]
        ws[i] = X @ betas[i]
    return betas, mean_betas

def ultimate_ar_bcd(X, y, eta_0, epsilon, lam=1, num_iters=10, intermediate_iter=np.ones(10)):
    """
    This method returns the ultimate matching pursuit with an inner loop solved using the AR-CBD technique.
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param eta_0: starting point (np.ndarray)
    :param epsilon: regulrization term (avoid to divide by 0)
    :param lam: regularization parameter
    :param num_iters: iteration number of the outer loop
    :param intermediate_iter: iteration number of the inner loop
    :return: sequences produces by the method
    """
    n, d = X.shape
    L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    # Perform a QR decomposition
    Q, R = np.linalg.qr(X.T, 'complete')
    assert np.allclose(X.T, np.dot(Q, R)), 'this is not equal!!'
    assert len(intermediate_iter) == num_iters, 'the number of intermediate_iter is not equal to the num_iters'
    Q2 = Q[:, n:]
    r = Q2.shape[1]
    # store the variables
    ws = np.zeros((num_iters, n))
    gammas = np.zeros((num_iters, d))
    etas = np.zeros((num_iters, d))
    zs = np.zeros((num_iters, r))
    thetas = np.zeros((num_iters, d))
    # Initialization
    etas[0] = eta_0
    ws[0] = X @ eta_0
    gammas[0] = np.ones(d) / d  # starting from a point in the simplex
    for i in range(1, num_iters):
        # compute the minimal value for z with an LMO
        grad = 1 / n * (ws[i - 1] - y)
        # CHAMBOLLE POCK FOR AN INTERMEDIATE ITERATION NUMBER
        etas_am, gammas_am, zs_am, thetas_am = ar_bcd(X=X,
                                                lam=lam,
                                                alpha=etas[i - 1],
                                                Q2=Q2,
                                                grad=grad,
                                                eta_0 = etas[i - 1],
                                                gamma_0=gammas[i - 1],
                                                L=L,
                                                p_1=0.5,
                                                p_2=0.5,
                                                intermediate_iters=intermediate_iter[i],
                                                epsilon=epsilon)
        etas[i] = etas_am[-1]
        gammas[i] = gammas_am[-1]
        zs[i] = zs_am[-1]
        thetas[i] = thetas_am[-1]
        ws[i] = X @ etas[i]
    return etas, gammas, zs, thetas

def ultimate_alternating_minization(X, y, eta_0, epsilon, lam=1, num_iters=10, intermediate_iter=np.ones(10)):
    """
    This method returns the ultimate matching pursuit with an inner loop solved using the alternating minimization technique.
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param eta_0: starting point (np.ndarray)
    :param epsilon: regularization term (avoids to divide by 0)
    :param lam: regularization paramter
    :param num_iters: iteration number for the outer loop
    :param intermediate_iter: iteration number of the inner loop
    :return: sequences produced by the method
    """
    n, d = X.shape
    L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    # Perform a QR decomposition
    Q, R = np.linalg.qr(X.T, 'complete')
    assert np.allclose(X.T, np.dot(Q, R)), 'this is not equal!!'
    assert len(intermediate_iter) == num_iters, 'the number of intermediate_iter is not equal to the num_iters'
    Q2 = Q[:, n:]
    r = Q2.shape[1]
    # store tghe variables
    ws = np.zeros((num_iters, n))
    gammas = np.zeros((num_iters, d))
    etas = np.zeros((num_iters, d))
    zs = np.zeros((num_iters, r))
    # Initialization
    etas[0] = eta_0
    ws[0] = X @ eta_0
    gammas[0] = np.ones(d) / d  # starting from a point in the simplex
    for i in range(1, num_iters):
        # compute the minimal value for z with an LMO
        grad = 1 / n * (ws[i - 1] - y)
        # CHAMBOLLE POCK FOR AN INTERMEDIATE ITERATION NUMBER
        etas_am, gammas_am, zs_am = alternating_minimization(X=X,
                                                lam=lam,
                                                alpha=etas[i - 1],
                                                Q2=Q2,
                                                grad=grad,
                                                eta_0=etas[i - 1],
                                                gamma_0=gammas[i - 1],
                                                L=L,
                                                intermediate_iters=intermediate_iter[i],
                                                epsilon=epsilon)
        etas[i] = etas_am[-1]
        gammas[i] = gammas_am[-1]
        zs[i] = zs_am[-1]
        ws[i] = X @ etas[i]
    return etas, gammas, zs

## A fully-corrective technique and the ultimate corrective matching pursuit

def fully_corrective(X, y, w_0, lam=1, num_iters=10, intermediate_iter=50, L=0):
    """
    The method returns the iterates generated by the fully-corrective matching pursuit.
    Given the LMO in the gradient, the method optimizes fully the set of active atoms (already selected atoms)
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param w_0: starting point (np.ndarray)
    :param lam: regularization parameter
    :param num_iters: iteration number of the outer loop
    :param intermediate_iter: iteration number of the inner loop
    :param L: Lipschitz constant (0 if not already computed)
    :return: sequence produced by the method
    """
    n, d = X.shape
    if L == 0:
        L = max(np.linalg.eigvalsh(X.T @ X)) / n
    gamma = 1/L
    # Compute the set of nonzero values for U for the initialization (with projected set)
    ws = np.zeros((num_iters, d))
    ws[0] = w_0

    # Compute the oracle and z_min
    S = []
    len_S = np.arange(num_iters)
    nonzero = np.arange(num_iters)
    for i in range(1, num_iters):
        # compute the minimal value for z with an LMO
        grad = 1 / n * X.T @ (X @ ws[i - 1] - y)
        # compute the LMO
        grad_idx = -np.max(np.abs(grad))
        id_grad = np.argmax(np.abs(grad))
        if max(0, - grad_idx - lam) > 0:
            if id_grad not in S:
                S.append(id_grad)
            len_S[i] = len(S)
            # Perform proximal gradient
            X_S = X[:, S]
            L_S = max(np.linalg.eigvalsh(X_S.T @ X_S)) / n
            gamma_S = 1/L_S
            w_int = proximal_gradient_descent_lasso(X_S, y, ws[i-1][S], gamma_S, lam=lam, num_iters=intermediate_iter, ord=1)[-1]
            ws[i][S] = w_int
            nonzero[i] = np.count_nonzero(w_int)
        else:
            print('We have found the support, and can optimize over it!')
            w_int = proximal_gradient_descent_lasso(X_S, y, ws[i-1][S], gamma_S, lam=lam, num_iters=num_iters, ord=1)[-1]
            for k in range(i, num_iters):
                ws[k][S] = w_int
            return ws
    return ws

def ultimate_corrective_method(X, y, w_0, lam=1, num_iters=10):
    """
    This method returns an alternative to the ultimate corrective matching pursuit.
    Given the LMO, the method fully optimize the quadratic upper bound given the active set of atoms.
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param w_0: startig point (np.ndarray)
    :param lam: regularization parameter
    :param num_iters: iteration number
    :return: sequence produced by the method
    """
    n, d = X.shape
    ws = np.zeros((num_iters, n))
    # Compute the oracle and z_min
    S = []
    len_S = np.zeros(num_iters)
    betas = np.zeros((num_iters, d))
    ws[0] = w_0
    L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
    # Compute the oracle and z_min
    for i in range(1, num_iters):
        # compute the minimal value for z with an LMO
        grad = 1 / n * (ws[i - 1] - y)
        grad_idx = -np.max(np.abs(X.T @ grad))
        # print(np.sort(np.abs(grad))[-10:])
        id_grad = np.argmax(np.abs(X.T @ grad))
        if max(0, - grad_idx - lam) > 0:
            #print('continue to add points', i, len(S), grad_idx, lam, id_grad)
            if id_grad not in S:
                S.append(id_grad)
            s = len(S)
            len_S[i] = len(S)
            X_S = X[:, S]
            L_S = max([np.linalg.norm(X[:, i]) for i in S]) ** 2 / n
            # Perform a verification with cvxpy
            eta = cp.Variable(s)
            beta = cp.Variable(s)
            constraints = [X_S @ eta == X_S @ (beta - betas[i - 1][S]) + ws[i - 1]]
            objective = cp.Minimize(grad.T @ (X_S @ (beta - betas[i - 1][S])) + L / 2 * cp.norm(beta - betas[i - 1][S], 1) ** (2) + lam * cp.norm(eta, 1))
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_BASIS_TOL_X': 1.0e-9})
            ws[i] = X_S @ beta.value
            betas[i][S] = eta.value
        else:
            print('We have found the support, and can optimize over it!')
            for k in range(i, num_iters):
                betas[k][S] = betas[i][S]
            return betas
    return betas























