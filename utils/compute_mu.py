import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from itertools import *
np.random.seed(0)


def compute_real_mu1(d, n, X):
    """
    Compute the real value for mu_1^L.
    WARNING: this computation is exponentional in the dimension.
    :param d: dimension
    :param n: number of samples
    :param X: input data (np.ndarray)
    :return:
    """
    # compute the extreme points
    s = np.array(list(product([-1, 1], repeat=n)))
    assert s.shape[0] == 2 ** n
    J = np.array(list(combinations(np.arange(d), n)))
    assert J.shape[0] == math.comb(d, n)
    # Find manually the maximum point
    entries = []
    for i in range(len(J)):
        for j in range(len(s)):
            inv_X_j = np.linalg.inv(X[:, J[i]])
            if np.linalg.norm(X.T @ inv_X_j.T @ s[j], np.inf) <= 1.:
                entries.append(np.linalg.norm(inv_X_j.T @ s[j]))
    return 1/np.max(entries)

def compute_real_mu_str(P):
    """
    Compute the real value for mu_1 for small values of dimaneison d and number of samples n.
    WARNING: this computation is exponential in the number of samples.
    :param P: input data (np.ndarray)
    :return:
    """
    n, d = P.shape
    Q = P @ np.linalg.inv(P.T @ P)
    x = np.array(list(product([-1, 1], repeat=d)))
    maxi = np.max(np.linalg.norm(Q @ x.T, axis=1))
    return (1 / np.max(np.linalg.norm(Q @ x.T, axis=0))) ** 2

def compute_mu1_underparametrized(P):
    """
    In the underparametrized regime (n >= d), this function returns the SDP approximation for mu_1.
    :param P: input data (np.ndarray)
    :return:
    """
    n, d = np.shape(P)
    assert n >= d, f'n={n} <= d={d}: this is not the underparametrized regime'
    #print('I compute mu_1 in the underparametrized regime')
    C = np.linalg.inv(P.T @ P)
    X = cp.Variable((d, d))
    constraints = [ X >> 0]
    constraints = constraints + [cp.diag(X) <= 1.]
    objective = cp.Maximize(cp.trace(C @ X))
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return 1/prob.value

def compute_mu1_overparametrized(X, n_max=20, d_max=400):
    """
    In the overparametrized regime (d >= n), this function returns the SDP approximation for mu_1^L for small values
    of dimension d_max and number of samples n_max (otherwise, the SDP may be long to compute). For large dimensions
    and number of samples, this function returns the lower bound mu_2^L/d.
    :param X: input data (np.ndarray)
    :param n_max: maximal size for the SDP.
    :return:
    """
    n, d = X.shape
    assert n <= d, f'n={n} >= d={d}, this is not the overparametrized regime'
    #print('I compute mu_1 in the overparametrized regime')
    if d<= d_max and n <= n_max:
        M = cp.Variable((n, n))
        constraints = [M >> 0]
        objective = cp.Minimize(cp.norm(cp.diag(X.T @ M @ X), "inf"))
        constraints += [cp.trace(M) == 1.]
        prob = cp.Problem(objective, constraints)
        mu_1 = prob.solve()
    else:
        mu_1 = (1 - np.sqrt(n / d)) ** 2
    return mu_1

