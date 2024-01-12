"""
This file returns the convergence in function value run of the following methods on a LASSO problem
- the ultimate matching pursuit,
- the regularized matching pursuit,
- the proximal gradient method,
- coordinate descent with the Gauss-Southwell rule.
The LASSO model is generated from i.i.d. gaussian random variables.
We plot the convergence in function values, and the behavior of the coordinates with a focus on the first iterates.
"""

import numpy as np
import matplotlib.pyplot as plt

from algorithms.proximal_gradient_descent import proximal_gradient_descent_lasso
from algorithms.coordinate_descent import gsq_cd_lasso
from algorithms.ultimate_method import ultimate_method
from algorithms.regularized_matching_pursuit import regularized_matching_pursuit

from generate_data.sparse_optimum import generate_sparse_optimum
from generate_data.gaussian import generate_uniform_gaussian_data

from utils.compute_mu import compute_mu1_overparametrized

#random seed
np.random.seed(0)

## PARAMETERS
d = 500 # dimension
n = 50 # number of samples
s = 8 # sparsity level
sigma = 0.5 # random noise

## Generate Gaussian random data
w_ = generate_sparse_optimum(d, s)
y, X, E = generate_uniform_gaussian_data(d, n, w_, sigma)
lam = 0.2 # regularization parameter
# Compute strong-convexity and smoothness parameters (in the ||.||_2 and the ||.||_1 norm)
L = max(np.linalg.eigvalsh(X.T @ X)) / n
L_1 = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
mu1 = compute_mu1_overparametrized(X)
gamma = 1/L

## OPTIMIZATION METHOD
num_iters = 500 # iteration number
full_iters = int(1.2 * num_iters) # extended iteration number
# starting point
w_0 = np.zeros(d)
methods = {'prox_grad': [], 'rmp': [], 'gsq': [], 'ultimate': []}

ws = np.zeros((full_iters, d))
u = 0
for method in list(methods.keys()):
    ws = np.zeros((full_iters, d))
    if method == 'ultimate':
        w_n = np.zeros(n)
        ws, _ = ultimate_method(X=X, y=y, w_0=w_n, lam=lam, num_iters=full_iters)
    if method == 'prox_grad':
        ws = proximal_gradient_descent_lasso(X, y, w_0, gamma, lam, full_iters)
    if method == 'rmp':
        ws = regularized_matching_pursuit(X, y, w_0, lam, full_iters, L=L_1)
    if method == 'gsq':
        ws = gsq_cd_lasso(X, y, w_0, lam, full_iters, lipschitz=False, L=L)
    methods[method].append(ws)

## COMPUTE THE OPTIUMUM IN FUNCTION VALUES
f_opts = []
for method in list(methods.keys()):
    f_opts.append(.5 / n * np.linalg.norm(X @ methods[method][0][-1] - y) ** 2 + lam * np.linalg.norm(
                methods[method][0][-1], 1))
f = min(f_opts)

## PLOT CONVERGENCE IN FUNCTION VALUES
color = {'ultimate': 'blue', 'gsq': 'green', 'rmp': 'red', 'prox_grad': 'orange'}

fig, axs = plt.subplots(1, 3, figsize=(23, 5))
fs = np.zeros((len(methods), num_iters))
u = 0
for method in list(methods.keys()):
    for i in range(num_iters):
        fs[u][i] = .5 / n * np.linalg.norm(X @ methods[method][0][i] - y) ** 2 + lam * np.linalg.norm(
                methods[method][0][i], 1)
    if method =='gsq':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method], label='CD with GS rule')
    elif method =='prox_grad':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method], label='proximal gradient')
    elif method =='rmp':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method], label='regularized MP')
    elif method =='ultimate':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method], label='ultimate')
        axs[0].plot(np.arange(num_iters), (1 - mu1/L_1/n)**(2 * np.arange(num_iters)), '--', color=color[method])
    u += 1

axs[0].semilogy()
axs[0].legend(fontsize=16)
axs[0].set_xlabel('iteration number',  fontsize=18)
axs[0].set_ylabel('f(x_k) - f_*',  fontsize=18)

axs[0].set_ylim(10 ** -9, 1)
axs[0].tick_params(axis='both', labelsize=18)

## PLOT COORDINATES
small_iters = 60
for method in list(methods.keys()):
    for i in range(d):
        axs[1].plot(np.arange(num_iters), methods[method][0][:num_iters, i], alpha=0.5, color=color[method])
        axs[2].plot(np.arange(small_iters), methods[method][0][:small_iters, i], alpha=0.5, color=color[method])
for u in range(2):
    axs[u+1].set_xlabel('iteration number', fontsize=18)
    axs[u+1].set_ylabel('coordinates', fontsize=18)
    axs[u+1].tick_params(axis='both', labelsize=18)
fig.suptitle(f'Convergence for d={d}, n={n}, s={s}, lambda={lam}', fontsize=18)

plt.show()