"""
This file compares the convergence guarantees for gradient descent and steepest coordinate descent for
a linear regression model. The data are:
- synthetic i.i.d. random variables,
- extracted from the Leukemia dataset.
Convergence bounds are compared to SDP approximations for mu_1 and mu_1^L.
The number of samples n is fixed and the experiment is performed for different dimensions d.
"""

import numpy as np
import matplotlib.pyplot as plt

from algorithms.proximal_gradient_descent import proximal_gradient_descent_lasso
from algorithms.regularized_matching_pursuit import regularized_matching_pursuit

from generate_data.sparse_optimum import generate_sparse_optimum
from generate_data.gaussian import generate_uniform_gaussian_data
from generate_data.real_datasets import load_leukemia, preprocess_standard_scale

from utils.compute_mu import compute_mu1_underparametrized, compute_mu1_overparametrized

np.random.seed(5)

models = ['synthetic', 'leukemia']

## PARAMETERS
# iteration number
num_iters = 1000
full_iters = int(1.2 * num_iters)

# Dimensions for Gaussian datas and RF (equivalent to ms)
dims = np.array([10, 15, 20, 50, 100, 200])
dims_max = max(dims)
color_dims = plt.cm.plasma(np.linspace(0, 1, len(dims)))
print('dimensions: ', dims, 'with maximal dimension', dims_max)
D = len(dims)

## DATASETS
# Gaussian dataset
sigma = 0.5
n_gauss = 20
s = 8
# Load the leukemia dataset
X_leuk, y_leuk = load_leukemia()
# Choice of models and optimization methods
models = ['synthetic', 'leukemia'] # synthetic correspond to i.i.d. gaussian data

## OPTIMIZATION METHODS
# For linear regression, the proximal gradient corresponds to gradient descent, and regularized matching pursuit ('rmp'),
# corresponds to steepest coordinate descent.
methods = {'prox_grad': [], 'rmp': []}

# Store the iterates, their optima and smoothness / strong-convexity parameters
fs, fs_opt = np.zeros((len(models), D, len(methods), num_iters)), np.zeros((len(models), D))
L2s, L1s = np.zeros((len(models), D)), np.zeros((len(models), D))
mu2s, mu2s_L= np.zeros((len(models), D)), np.zeros((len(models), D))
mu1s, mu1s_L = np.zeros((len(models), D)), np.zeros((len(models), D))

# verify with the models
ns = np.zeros((len(models), D))

# Compute for each dimension the models, and performs gradient descent / steepest coordinate descent.
for m in range(len(models)):
    for i in range(D):
        # methods to test
        methods = {'prox_grad': [], 'rmp': []}
        ws = np.zeros((len(methods), full_iters, dims[i]))
        ## generate the data depending on the dimension
        d = dims[i]
        if models[m] == 'synthetic':
            ns[m, i] = n_gauss
            ns[m, i].astype(int)
            w_ = generate_sparse_optimum(d=dims[i], s=s)
            y, X, E = generate_uniform_gaussian_data(d=dims[i], n=int(ns[m, i]), w_=w_, sigma=sigma)
        if models[m] == 'leukemia':
            X = preprocess_standard_scale(X_leuk, dims[i])
            y = y_leuk
            ns[m, i] = X.shape[0]
        # Compute smoothness and strong convexity parameters
        L2s[m][i] = max(np.linalg.eigvalsh(X.T @ X)) / ns[m, i]
        L1s[m][i] = max([np.linalg.norm(X[:, j]) ** 2 for j in range(dims[i])]) / ns[m, i]
        mu2s[m][i] = max(0, min(np.linalg.eigvalsh(X.T @ X)) / ns[m, i])
        if models[m] == 'leukemia':
            mu2s_L[m][i] = np.linalg.eigvalsh((X @ X.T)/ns[m, i])[1] # due to the preprocessing step (standard scaler)
        else:
            mu2s_L[m][i] = max(0, min(np.linalg.eigvalsh((X @ X.T)/ns[m, i])))
        if dims[i] <= ns[m, i]:
            mu1s[m][i] = compute_mu1_underparametrized(X)
        else:
            mu1s_L[m][i] = compute_mu1_overparametrized(X)
        # perform the optimization method
        w_0 = np.zeros(dims[i]) # initial point
        u = 0
        f_opt = []
        for method in list(methods.keys()):
            if method == 'prox_grad':
                ws[u] = proximal_gradient_descent_lasso(X, y, w_0, 1/L2s[m][i], 0, full_iters)
            if method == 'rmp':
                ws[u] = regularized_matching_pursuit(X, y, w_0, 0, full_iters, L=L1s[m][i])
            for k in range(num_iters):
                fs[m][i][u][k] = .5 / ns[m, i] * np.linalg.norm(X @ ws[u][k] - y) ** 2
            f_opt.append(.5 / ns[m, i] * np.linalg.norm(X @ ws[u][-1] - y) ** 2)
            u += 1
        fs_opt[m][i] = min(f_opt)
        fs[m][i] = (fs[m][i] - fs_opt[m][i])

fig, axs = plt.subplots(len(methods), len(models), figsize=(15, 15))

for m in range(len(models)):
    u = 0
    for method in list(methods.keys()):
        for i in range(D):
            axs[u, m].plot(np.arange(num_iters), fs[m][i][u] / fs[m][i][u][0], color=color_dims[i],
                           label=f'd={dims[i]}')
            if method == 'prox_grad':
                axs[u, m].plot(np.arange(num_iters), (1 - max(mu2s[m][i], mu2s_L[m][i]) / L2s[m][i]) ** (2 * np.arange(num_iters)), '--',
                               color=color_dims[i])
            elif method == 'rmp':
                axs[u, m].plot(np.arange(num_iters),
                               (1 - max(mu1s[m][i], mu1s_L[m][i])/ L1s[m][i] / ns[m, i]) ** (2 * np.arange(num_iters)), '--',
                               color=color_dims[i])

        axs[u, m].set_xlabel('iteration number', fontsize=18)
        axs[u, m].set_ylabel('f(x_k) - f_*', fontsize=18)
        axs[u, m].set_ylim(10 ** -9, 1)
        axs[u, m].set_yscale('log')
        axs[u, m].tick_params(axis='both', labelsize=18)
        if method == 'prox_grad':
            axs[u, m].set_title(f'gradient descent for {models[m]}', fontsize=18)
        elif method == 'rmp':
            axs[u, m].set_title(f'CD with GS rule for {models[m]}', fontsize=18)
        if u == 0 and m == 0:
            axs[u, m].legend(fontsize=15)
        u += 1

plt.show()