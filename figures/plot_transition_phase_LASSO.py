"""
This file compares epsilon curves for the proxiaml gradient descent and proximal coordinate descent with
the GS-rule for the LASSO. That is, it plots the level lines for iteration numbers at which a certain precision
epsilon is achieved by the optimization method. This let a transition phase appear between underparametrized and
overparametrized models.
Data are generated from:
- synthetic i.i.d. random variables,
- the Leukemia dataset,
- a random feature model constructed from the Leukemia dataset.
The number of samples n is fixed and the experiment is performed for different dimensions d.
"""

import numpy as np
import matplotlib.pyplot as plt

from algorithms.proximal_gradient_descent import proximal_gradient_descent_lasso
from algorithms.coordinate_descent import gsq_cd_lasso
from algorithms.regularized_matching_pursuit import regularized_matching_pursuit

from generate_data.sparse_optimum import generate_sparse_optimum
from generate_data.gaussian import generate_uniform_gaussian_data

from utils.epsilon_curve import compute_epsilon_curve

np.random.seed(0)

## PARAMETERS
d = 500 # dimension d
n = 50 # number of samples
s = 8 # sparsity level
sigma = 0.5 # noise

## RANDOM GAUSSIAN DATA MODEL
w_ = generate_sparse_optimum(d, s)
print('optimal point', w_)
y, X, E = generate_uniform_gaussian_data(d, n, w_, sigma)
L = max(np.linalg.eigvalsh(X.T @ X)) / n # smoothness parameter in norm ||.||_2
L_1 = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n # smoothness parameter in norm ||.||_1
if sigma != 0:
    print('ratio signal to noise', np.linalg.norm(X @ w_) / np.linalg.norm(X @ w_ - y))
# LASSO modeled by the regularization parameter.
lambdas = np.concatenate((np.zeros(1), np.logspace(-7, 0, 20)), axis=0)

## OPTIMIZATION METHODS, where 'rmp' corresponds to regularization matching pursuit
methods = {'prox_grad': [], 'gsq': [], 'rmp': []}
gamma = 1 / L
# starting point
w_0 = np.zeros(d)
num_iters = 100 # iteration number
full_iters = int(1.2 * num_iters) # extended iteration number

# perform the optimization methods
for method in list(methods.keys()):
    ws = np.zeros((len(lambdas), full_iters, d))
    u = 0
    print(method)
    for k in range(len(lambdas)):
        print(lambdas[k])
        if method == 'prox_grad':
            ws[k] = proximal_gradient_descent_lasso(X, y, w_0, gamma, lambdas[k], full_iters)
        if method == 'rmp':
            ws[k] = regularized_matching_pursuit(X, y, w_0, lambdas[k], full_iters, L=L_1)
        if method == 'gsq':
            ws[k] = gsq_cd_lasso(X, y, w_0, lambdas[k], full_iters, lipschitz=False, L=L)
    methods[method].append(ws)
# Compute the difference in function values
datas = np.zeros((len(lambdas), len(methods), num_iters))
f_opts = np.zeros((len(lambdas), len(methods)))
fs = np.zeros((len(lambdas), len(methods), num_iters))

for k in range(len(lambdas)):
    u = 0
    for method in list(methods.keys()):
        f_opts[k][u] = .5 / n * np.linalg.norm(X @ methods[method][0][k][-1] - y) ** 2 + lambdas[k] * np.linalg.norm(
            methods[method][0][k][-1], 1)
        u += 1
    f = min(f_opts[k])
    u = 0
    for method in list(methods.keys()):
        for i in range(num_iters):
            fs[k][u][i] = .5 / n * np.linalg.norm(X @ methods[method][0][k][i] - y) ** 2 + lambdas[k] * np.linalg.norm(
                methods[method][0][k][i], 1)
            datas[k][u][i] = fs[k][u][i] - f # store the distance in function value
        u += 1

## EPSILON CURVES
N_error = 15
ks = np.zeros((len(methods), len(lambdas), N_error+1))
for u in range(len(methods)):
    ks[u], epsilon = compute_epsilon_curve(N_error, datas[:, u, :])

## PLOT THE EPSILON FIT
fig, axs = plt.subplots(1, 3, figsize=(19, 6))
log = '-log'  # choose 'symlog' or 'log' or '-log'
colors = np.flip(plt.cm.plasma(np.linspace(0, 1, len(epsilon))), axis=0)

u = 0
for method in list(methods.keys()):
    for e in range(len(epsilon)):
        if log == '-log':
            if e==0:
                axs[u].plot(-np.log(np.array(lambdas)), ks[u, :, e], label=f'e=1e{e}', color=colors[e])
            else:
                axs[u].plot(-np.log(np.array(lambdas)), ks[u, :, e], label=f'e=1e-{e}', color=colors[e])
        else:
            if e == 0:
                axs[u].plot(lambdas, ks[u, :, e], label=f'e=1e{e}', color=colors[e])
            else:
                axs[u].plot(lambdas, ks[u, :, e], label=f'e=1e-{e}', color=colors[e])
    axs[u].set_xlabel('lambda', fontsize=19)
    axs[u].set_ylabel('k(e)', fontsize=19)
    if method == 'prox_grad':
        axs[u].set_title('proximal gradient', fontsize=18)
    elif method == 'rmp':
        axs[u].set_title('regularized matching pursuit', fontsize=18)
    elif method == 'gsq':
        axs[u].set_title('coordinate descent with GS rule', fontsize=18)
    if log == 'log':
        axs[u].set_xscale('log')
        axs[u].set_yscale('log')
        axs[u].set_xlabel('lambda', fontsize=18)
    elif log == 'symlog':
        axs[u].set_xscale('symlog')
        axs[u].set_yscale('symlog')
        axs[u].set_xlabel('lambda', fontsize=18)
    elif log == '-log':
        axs[u].set_yscale('log')
        axs[u].set_xlabel('log(1/lambda)', fontsize=18)
    axs[u].set_ylim(0, 5 * 10 ** 4)
    axs[u].tick_params(axis='both', labelsize=18)
    u += 1
fig.suptitle(f'Epsilon fit for d={d}, n={n}, s={s}, sigma={sigma}', fontsize=18)
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

plt.show()