"""
This file returns the plot of the convergence behavior for the ultimate matching pursuit on a LASSO problem,
computed using several inner loop strategies:
- computing the inner loop with cvxpy (MOSEK solver)
- the Chambolle-Pock algorithm,
- an alternating minimization technique,
- the AR_BCD technique (alternating-minimization and randomized block coordinate descent.
The LASSO model is generated from i.i.d. gaussian random variables.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from algorithms.regularized_matching_pursuit import regularized_matching_pursuit
from algorithms.ultimate_method import ultimate_method, ultimate_ar_bcd, ultimate_alternating_minization


from generate_data.gaussian import generate_uniform_gaussian_data
from generate_data.sparse_optimum import generate_sparse_optimum

from utils.compute_mu import compute_mu1_overparametrized

#random seed
np.random.seed(0)

## PARAMETERS
d = 50 # dimension
n = 20 # number of samples
s = 8 # sparsity level
sigma = 0.5 # random noise

## Generate the LASSO model
w_ = generate_sparse_optimum(d, s)
y, X, E = generate_uniform_gaussian_data(d, n, w_, sigma)
lam = 0.001 # regularization parameter
# Compute strong-convexity and smoothness parameters (in the ||.||_2 and the ||.||_1 norm)
L = max(np.linalg.eigvalsh(X.T @ X)) / n
L_1 = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n
mu_1 = compute_mu1_overparametrized(X)/n

## METHODS TO PLOT
methods = {'rmp': [], 'ultimate_ar_bcd': [], 'ultimate_am': [], 'ultimate': []}
gamma = 1 / L
# Iteration number
num_iters = 100
full_iters = int(1.1 * num_iters)
# Inner loop strategy
inner_loop = 'best' # or 'linear' or 'exponential'
multi_inner = 8
if inner_loop == 'exponential':
    r = 1 - mu_1 / L_1
    print('convergence guarantee', r ** 2)
    int_iters = 1 / (r ** 2) ** np.arange(num_iters)
    int_iters = multi_inner * int_iters.astype(int) + 1
    print('last iteration number of the inner loop', int_iters[-1])
elif inner_loop == 'linear':
    int_iters = multi_inner * np.arange(num_iters)
elif inner_loop == 'best':
    r = 1 - mu_1 / L_1
    print('convergence guarantee', r ** 2)
    int_iters = (1 / (r ** 2)) ** np.arange(num_iters)
    int_iters = int_iters.astype(int) + 1
    int_iters = multi_inner * np.array([max(int_iters[i], i) for i in range(num_iters)])
    print('last iteration number of the inner loop', int_iters[-1])

# starting point
w_0 = np.zeros(d)
ws = np.zeros((full_iters, d))
fs = np.zeros((len(methods), num_iters))
f_opts = []

u = 0
for method in list(methods.keys()):
    ws = np.zeros((full_iters, d))
    print(method)
    start_time = time.time()
    if method == 'ultimate':
        w_n = np.zeros(n)
        ws, _ = ultimate_method(X=X, y=y, w_0=w_n, lam=lam, num_iters=full_iters)
    if method == 'rmp':
        ws = regularized_matching_pursuit(X, y, w_0, lam, full_iters, L=L_1)
    if method == 'ultimate_am':
        # constant noise to help evaluate AM
        epsilon = 10 ** -7
        ws, _, _ = ultimate_alternating_minization(X=X, y=y, eta_0=w_0, epsilon=epsilon, lam=lam, num_iters=num_iters,
                                                   intermediate_iter=int_iters)
    if method == 'ultimate_ar_bcd':
        # constant noise to help evaluate AM
        epsilon = 10 ** -7
        ws, _, _, _ = ultimate_ar_bcd(X=X, y=y, eta_0=w_0, epsilon=epsilon, lam=lam, num_iters=num_iters,
                                      intermediate_iter=int_iters)
    print("--- %s seconds ---" % (time.time() - start_time))
    methods[method].append(ws)

    # Compute the optimum
    f_opts.append(
        .5 / n * np.linalg.norm(X @ methods[method][0][-1] - y) ** 2 + lam * np.linalg.norm(methods[method][0][-1], 1))
    ws_cp = np.cumsum(ws, axis=0)
    fs_cp = np.zeros(num_iters)
    for i in range(num_iters):
        fs[u][i] = .5 / n * np.linalg.norm(X @ methods[method][0][i] - y) ** 2 + lam * np.linalg.norm(
            methods[method][0][i], 1)
    u += 1
# Compute the minimum of the different methods
f = min(f_opts)
u = 0
for method in list(methods.keys()):
    print(method, f_opts[u])
    u += 1

## PLOT THE FIGURE
color = {'ultimate': 'blue', 'rmp': 'cyan',  'ultimate_am': 'violet', 'ultimate_ar_bcd': 'green'}

## PLOT CONVERGENCE IN FUNCTION VALUES
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

u = 0
for method in list(methods.keys()):
    # plot convergence in function value
    if method == 'ultimate':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method], label=method)
    elif method == 'ultimate_am':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method],
                    label='alternating minimization')
    elif method == 'ultimate_ar_bcd':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method], label='AR-BCD')
    elif method == 'rmp':
        axs[0].plot(np.arange(num_iters), (fs[u] - f) / (fs[u][0] - f), color=color[method],
                    label='regularized MP')
    # plot the coordinates
    for i in range(d):
        axs[1].plot(np.arange(num_iters), methods[method][0][:num_iters, i], color=color[method], alpha=0.7)
    u += 1

axs[0].plot(np.arange(num_iters), (1 - mu_1 / L_1) ** (2 * np.arange(num_iters)), '--', color='blue', alpha=0.7)

axs[0].semilogy()
axs[0].legend(fontsize=18)
axs[0].set_xlabel('iteration number', fontsize=18)
axs[0].set_ylabel('f - f_*', fontsize=18)
axs[0].set_ylim(10 ** -8, 1.5)
axs[0].set_title(f'lambda={lam}, d={d}, s={s}, n={n}', fontsize=18)
axs[0].tick_params(axis='both', labelsize=18)

axs[1].set_xlabel('iteration number', fontsize=18)
axs[1].set_ylabel('coordinates', fontsize=18)
axs[1].tick_params(axis='both', labelsize=18)
plt.show()