"""
This file compares epsilon curves for gradient descent and steepest coordinate descent for a linear regression model.
That is, it plots the level lines for iteration numbers at which a certain precision
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
from algorithms.regularized_matching_pursuit import regularized_matching_pursuit

from generate_data.sparse_optimum import generate_sparse_optimum
from generate_data.gaussian import generate_uniform_gaussian_data
from generate_data.real_datasets import load_leukemia, preprocess_standard_scale
from generate_data.generate_random_features import return_RF

from utils.epsilon_curve import compute_epsilon_curve

np.random.seed(6)

# MODELS for the data
# We consider three data models for linear regression. 'synthetic' corresponds to i.i.d. gaussian variables.
models = ['synthetic', 'leukemia', 'random_features']

## PARAMETERS
num_iters = 10000 # iteration number
full_iters = int(1.2 * num_iters) # real iteration number to plot convergence with respect to a better optimum.
# Gaussian dataset
sigma = 0.5
n_gauss = 50
s = 8

# DIMENSIONS
dims = np.logspace(1, 3, 20)
dims = dims.astype(int)
dims_max = max(dims)
D = len(dims)
# colors for dimensions d
color_dims = plt.cm.plasma(np.linspace(0, 1, len(dims)))

# RANDOM FEATURES
d_r = 200 # dimension of the dataset (and not of the random features)
theta = np.random.multivariate_normal(np.zeros(d_r), sigma * np.eye(d_r), size=dims_max)
assert theta.shape == (dims_max, d_r)
# Load the leukemia dataset
X_leuk, y_leuk = load_leukemia()


## OPTIMIZATION METHODS
methods = {'prox_grad': [], 'rmp': []}
fs, fs_opt = np.zeros((len(models), D, len(methods), num_iters)), np.zeros((len(models), D))
# verify with the models
ns = np.zeros((len(models), D))

for m in range(len(models)):
    print(models[m])
    for i in range(D):
        # methods to test
        methods = {'prox_grad': [], 'rmp': []}
        ws = np.zeros((len(methods), full_iters, dims[i]))
        ## compute the datamodel
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
        if models[m] == 'random_features':
            X_r = preprocess_standard_scale(X_leuk, d_r)
            X = return_RF(X_r, theta, dims[i])
            y = y_leuk
            ns[m, i] = X.shape[0]
        # initial point
        w_0 = np.zeros(dims[i])
        u = 0
        f_opt = []
        L2 = max(np.linalg.eigvalsh(X.T @ X)) / ns[m, i] # smoothness parameter in norm ||.||_2
        L1 = max([np.linalg.norm(X[:, j]) ** 2 for j in range(dims[i])]) / ns[m, i] # smoothness parameter in norm ||.||_1
        for method in list(methods.keys()):
            if method == 'prox_grad':
                ws[u] = proximal_gradient_descent_lasso(X, y, w_0, 1 / L2, 0, full_iters)
            if method == 'rmp':
                ws[u] = regularized_matching_pursuit(X, y, w_0,0, full_iters, L=L1)
            for k in range(num_iters):
                fs[m][i][u][k] = .5 / ns[m, i] * np.linalg.norm(X @ ws[u][k] - y) ** 2
            f_opt.append(.5 / ns[m, i] * np.linalg.norm(X @ ws[u][-1] - y) ** 2)
            u += 1
        fs_opt[m][i] = min(f_opt)
        fs[m][i] = (fs[m][i] - fs_opt[m][i])
# Prepare a dataset to plot
datas = np.zeros((len(models), D, len(methods), num_iters))
for m in range(len(models)):
    for d in range(D):
        for u in range(len(methods)):
            for k in range(num_iters):
                datas[m, d, u, k] = fs[m, d, u, k] / fs[m, d, u, 0]

## PLOT THE FIGURE
fig, axs = plt.subplots(len(methods), len(models), figsize=(20, 14))
N_error = 10
colors_error = np.flip(plt.cm.plasma(np.linspace(0, 1, N_error)), axis=0)
for m in range(len(models)):
    u = 0
    for method in list(methods.keys()):
        ks, epsilon = compute_epsilon_curve(N_error, datas[m, :, u, :])
        for e in range(len(epsilon)-1):
            if e == 0:
                axs[u, m].plot(dims, ks[:, e], color=colors_error[e], label=f'e=1e{e}')
            else:
                axs[u, m].plot(dims, ks[:, e], color=colors_error[e], label=f'e=1e-{e}')
        if models[m] == 'random_features':
            name_mod = 'random features'
        else:
            name_mod = models[m]
        if method == 'prox_grad':
            name_met = 'gradient descent'
        elif method == 'rmp':
            name_met = 'CD with GS rule'
        axs[u, m].set_title(f'{name_met} for {name_mod} data', fontsize=18)
        axs[u, m].set_xscale('log')
        axs[u, m].set_yscale('log')
        axs[u, m].set_xlabel('dimension', fontsize=18)
        axs[u, m].set_ylabel('k(e)', fontsize=18)
        axs[u, m].set_ylim(0, 5*10 ** 5)
        axs[u, m].tick_params(axis='both', labelsize=18)
        u += 1
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
plt.show()
