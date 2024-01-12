"""
This file returns a plot of the convergence in function values of the proximal gradient and the regularized
matching pursuit iterated generated on a LASSO problem, for different values of lambda.
It shows a dependency between the value for lambda and the sparsity level. Given the final sparsity,
we compare the convergence to approximate upper bounds.

The LASSO model is generated from a unique synthetic gaussian dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

from generate_data.sparse_optimum import generate_sparse_optimum
from generate_data.gaussian import generate_uniform_gaussian_data

from algorithms.proximal_gradient_descent import proximal_gradient_descent_lasso
from algorithms.regularized_matching_pursuit import regularized_matching_pursuit

from utils.compute_mu import compute_mu1_underparametrized

np.random.seed(0)

## PARAMETERS
d = 500 # dimension
n = 50 # number of samples
s = 8 # sparsity level
sigma = 0.5 # random noise

## Generate gaussian random data.
w_ = generate_sparse_optimum(d, s)
y, X, E = generate_uniform_gaussian_data(d, n, w_, sigma)
L = max(np.linalg.eigvalsh(X.T @ X)) / n # smoothness parameter with respect to the ||.||_2 norm
mu = max(0, min(np.linalg.eigvalsh(X.T @ X)) / n) # strong-convexity parameter with respect to the ||.||_2 norm
L_1 = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n # smoothness parameter with respect to the ||.||_1 norm
# Values for lambda (penalization parameter)
lambdas = [0, 0.001, 0.07, 0.1, 0.2, 0.4]

## OPTIMIZATION METHODS
w_0 = np.zeros(d) # starting point
num_iters = 5000 # iteration number
full_iters = int(1.2 * num_iters) # extended iteration number
methods = {'prox_grad': [], 'rmp': []}
color = {'prox_grad': 'orange', 'rmp': 'cyan'}
names = {'gsq': 'coordinate descent with GS rule', 'prox_grad': 'proximal gradient', 'rmp': 'regularized matching pursuit'}
gamma = 1 / L

for method in list(methods.keys()):
    ws = np.zeros((len(lambdas), full_iters, d))
    u = 0
    for k in range(len(lambdas)):
        if method == 'prox_grad':
            ws[k] = proximal_gradient_descent_lasso(X, y, w_0, gamma, lambdas[k], full_iters)
        if method == 'rmp':
            ws[k] =regularized_matching_pursuit(X, y, w_0, lambdas[k], full_iters, L=L_1)
    methods[method].append(ws)
for k in range(len(lambdas)):
    f_opts = np.zeros(len(methods))
    u = 0
    for method in list(methods.keys()):
        f_opts[u] = .5 / n * np.linalg.norm(X @ methods[method][0][k][-1] - y) ** 2 + lambdas[k] * np.linalg.norm(
            methods[method][0][k][-1], 1)
        u += 1
    f = min(f_opts)
    fs = np.zeros((len(methods), num_iters))
    u = 0
    for method in list(methods.keys()):
        for i in range(num_iters):
            fs[u][i] = .5 / n * np.linalg.norm(X @ methods[method][0][k][i] - y) ** 2 + lambdas[k] * np.linalg.norm(
                methods[method][0][k][i], 1)

mu_1s = np.zeros((len(methods), len(lambdas)))
mu_2s = np.zeros((len(methods), len(lambdas)))


fig, axs = plt.subplots(1, 2, figsize=(13, 5))
# color_lam = {0: 'blue', 0.00001: 'green', 0.001: 'red', 0.01: 'orange',
#     0.1: 'cyan', 0.2: 'purple', 0.3: 'black' , 0.5: 'yellow'}
color_lam = np.flip(plt.cm.plasma(np.linspace(0, 1, len(lambdas))), axis=0)

# An optimal value for each lambda (not the same optimization problem)
f_lam = np.zeros((len(lambdas), len(methods)))
f_opts = np.zeros(len(lambdas))

for k in range(len(lambdas)):
    u_opt = 0
    for method in list(methods.keys()):
        f_lam[k][u_opt] = .5 / n * np.linalg.norm(X @ methods[method][0][k][-1] - y) ** 2 + lambdas[k] * np.linalg.norm(
            methods[method][0][k][-1], 1)
        u_opt += 1
    f_opts[k] = min(f_lam[k])

u = 0
for method in list(methods.keys()):
    fs = np.zeros((len(lambdas), num_iters))
    for k in range(len(lambdas)):
        # Compute restricted strong convexity on the support
        S = np.nonzero(methods[method][0][k][-1])[0]
        mu_2s[u][k] = min(np.linalg.eigvalsh(X[:, S].T @ X[:, S])) / n
        # Compute mu1
        if len(S) <= n:
            mu_1s[u][k] = compute_mu1_underparametrized(X[:, S])/n

        # Plot the numerical convergence value and its expected bound
        for i in range(num_iters):
            fs[k][i] = .5 / n * np.linalg.norm(X @ methods[method][0][k][i] - y) ** 2 + lambdas[k] * np.linalg.norm(
                methods[method][0][k][i], 1)
        axs[u].plot(np.arange(num_iters), (fs[k] - f_opts[k]) / (fs[k][0] - f_opts[k]), color=color_lam[k],
                    label=lambdas[k])
        if method == 'prox_grad':
            axs[u].plot(np.arange(num_iters), (1 - mu_2s[u][k] / L) ** np.arange(num_iters), '--', color=color_lam[k])
        elif method == 'rmp':
            axs[u].plot(np.arange(num_iters), (1 - mu_1s[u][k] / L_1) ** np.arange(num_iters), '--', color=color_lam[k])

    axs[u].set_title(f'{names[method]}', fontsize=16)
    axs[u].set_xlabel('iteration number', fontsize=16)
    axs[u].set_ylabel('convergence in function values', fontsize=16)
    axs[u].set_yscale('log')
    if u == 1:
        axs[u].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    axs[u].set_ylim(10 ** -9, 1)
    axs[u].tick_params(axis='both', labelsize=15)
    plt.suptitle(f'Convergence for d={d}, n={n}, s={s}, sigma={sigma}', fontsize=16)
    u += 1

plt.show()