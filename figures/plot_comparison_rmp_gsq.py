"""
This file compared the convergence in function values of the following optimization methods on a LASSO problem, starting
either from zero, or from a random point:
- the proximal gradient method,
- the regularized matching pursuit,
- the proximal coordinate descent method with Gauss-Southwell rule.
The LASSO model is generated from i.i.d gaussian random data.

When starting from zero, coordinate descent with GS rule and the regularized matching pursuit almost always match.
Yet, when starting from a random point, the regularized matching pursuit benefits from a better start.
"""

import numpy as np
import matplotlib.pyplot as plt

from generate_data.sparse_optimum import generate_sparse_optimum
from generate_data.gaussian import generate_uniform_gaussian_data

from algorithms.proximal_gradient_descent import proximal_gradient_descent_lasso
from algorithms.coordinate_descent import gsq_cd_lasso
from algorithms.regularized_matching_pursuit import regularized_matching_pursuit

np.random.seed(0)

## PARAMETERS
d = 500 # dimension
n = 50 # number of samples
s = 8 # sparsity level
sigma = 0.5 # random noise
lam = 0.1 # regularization parameter

## Generate i.i.d. gaussian random data
w_ = generate_sparse_optimum(d, s)
y, X, E = generate_uniform_gaussian_data(d, n, w_, sigma)
# Compute smoothness with respect to the ||.||_2 and ||.||_1 norm
L = max(np.linalg.eigvalsh(X.T @ X)) / n
mu = max(0, min(np.linalg.eigvalsh(X.T @ X)) / n)
L_1 = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
gamma = 1 / L

## OPTIMIZATION METHODS
num_iters = 500 # iteration number
full_iters = int(1.2 * num_iters) # extended iteration number
# starting point
w_0 = 2 * np.random.rand(d) - 1

methods = {'prox_grad': [], 'rmp': [], 'gsq': []}
color = {'ultimate': 'cyan', 'gsq': 'green', 'prox_grad': 'red', 'rmp': 'blue'}
names = {'gsq': 'coordinate descent with GS rule', 'prox_grad': 'proximal gradient', 'rmp': 'regularized matching pursuit'}

for method in list(methods.keys()):
    ws = np.zeros((full_iters, d))
    u = 0
    print(method)
    if method == 'prox_grad':
        ws = proximal_gradient_descent_lasso(X, y, w_0, gamma, lam, full_iters)
    elif method == 'rmp':
        ws = regularized_matching_pursuit(X, y, w_0, lam, full_iters, L=L_1)
    elif method == 'gsq':
        ws = gsq_cd_lasso(X, y, w_0, lam, full_iters, lipschitz=False, L=L_1)
    methods[method].append(ws)
f_opts = np.zeros(len(methods))
u = 0
for method in list(methods.keys()):
    f_opts[u] = .5 / n * np.linalg.norm(X @ methods[method][0][-1] - y) ** 2 + lam * np.linalg.norm(
        methods[method][0][-1], 1)
    u += 1
f = min(f_opts)

## PLOT CONVERGENCE IN FUNCTION VALUES
fs = np.zeros((len(methods), num_iters))
u = 0
for method in list(methods.keys()):
    for i in range(num_iters):
        fs[u][i] = .5 / n * np.linalg.norm(X @ methods[method][0][i] - y) ** 2 + lam * np.linalg.norm(
            methods[method][0][i], 1)
    plt.plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method], label=names[method])
    u += 1

plt.xlabel('iteration number', fontsize=15)
plt.ylabel('f(x_k) - f_*', fontsize=15)
plt.title(f'Convergence for d={d}, n={n}, s={s}', fontsize=15)
plt.semilogy()
plt.legend(fontsize=12)
plt.ylim(10 ** -6, 1)

plt.show()