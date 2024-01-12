"""
This file compares the values of :
- mu_1 and mu_2 in the underparametrized regime,
- mu_1^L and mu_2^L in the overparametrized regime,
for gaussian i.i.d. data with specific diagonal variances.
"""

### IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from generate_data.compute_mu import compute_mu1_underparametrized, compute_mu1_overparametrized

np.random.seed(0)


### PARAMETERS
avg = 5 # average over a certain number of trials

n_dim = 20
## Select the dimensions and number of samples in the underparametrized and the overparametrized regime
# underparametrized regime
n_under = 50
ds_under = np.linspace(5, n_under, n_dim).astype(int)
# overparametrized regime
n_over = 20
ds_over = np.linspace(n_over, n_over * 10, n_dim).astype(int)

# Select the type of variance for i.i.d gaussian random variables
gaussian_type = ['uniform', 'non_uniform', 'one_small', 'one_big']
# Store the values for mu_1 and mu_2
mu2s = np.zeros((2, len(gaussian_type), n_dim, avg))
mu1s = np.zeros((2, len(gaussian_type), n_dim, avg))

for j in range(2):
    if j == 0:
        status = 'under'
        n = n_under
        ds = ds_under
    else:
        status = 'over'
        n = n_over
        ds = ds_over
    for l in range(len(gaussian_type)):
        for i in range(n_dim):
            for k in range(avg):
                if gaussian_type[l] == 'uniform':
                    I = np.eye(ds[i])
                elif gaussian_type[l] == 'non_uniform':
                    I = np.eye(ds[i]) / (1 + np.arange(ds[i]))
                elif gaussian_type[l] == 'one_small':
                    I = np.eye(ds[i])
                    I[2, 2] = 100
                elif gaussian_type[l] == 'one_big':
                    I = np.eye(ds[i])
                    I[2, 2] = 0.01
                X = np.random.multivariate_normal(np.zeros(ds[i]), I, n)
                if n >= ds[i]:
                    # underparametrized
                    mu1s[j][l][i][k] = compute_mu1_underparametrized(X)
                    mu2s[j][l][i][k] = min(np.linalg.eigvalsh(X.T @ X))
                else:
                    # overparametrized
                    mu1s[j][l][i][k] = compute_mu1_overparametrized(X, n_max=31)
                    mu2s[j][l][i][k] = min(np.linalg.eigvalsh(X @ X.T))

### PLOT THE FIGURE
fig, axs = plt.subplots(2, len(gaussian_type), figsize=(30, 15))
for j in range(2):
    if j == 0:
        status = 'under'
        n = n_under
        ds = ds_under
    else:
        status = 'over'
        n = n_over
        ds = ds_over
    for l in range(len(gaussian_type)):
        axs[j, l].plot(ds, np.mean(mu1s[j, l, :, :], axis=1), color='red', label='SDP')
        axs[j, l].plot(ds, np.mean(mu2s[j, l, :, :], axis=1), '--', color='blue', label='mu_2')
        if status == 'under':
            axs[j, l].plot(ds, np.mean(mu2s[j, l, :, :], axis=1)/ds, '--', color='green', label='mu_2/d')
        elif status == 'over':
            axs[j, l].plot(ds, np.mean(mu2s[j, l, :, :], axis=1)/ds, '--', color='green', label='muL_2/d')
        axs[j, l].set_xlabel('dimension', fontsize=18)
        if j == 0:
            axs[j, l].set_ylabel('mu', fontsize=18)
        else:
            axs[j, l].set_ylabel('mu_PL', fontsize=18)
        axs[j, l].set_yscale('log')
        if gaussian_type[l] == 'uniform':
            axs[j, l].set_title(f'uniform with n={n}', fontsize=18)
            axs[j, l].legend(fontsize=16)
        if gaussian_type[l] == 'non_uniform':
            axs[j, l].set_title(f'non uniform with n={n}', fontsize=18)
        if gaussian_type[l] == 'one_small':
            axs[j, l].set_title(f'one small diagonal element with n={n}', fontsize=18)
        if gaussian_type[l] == 'one_big':
            axs[j, l].set_title(f'one large diagonal element with n={n}', fontsize=18)
        axs[j, l].tick_params(axis='both', labelsize=18)
fig.suptitle(f'Comparison of SDP relaxations for mu with different variances for a gaussian model', fontsize=20)

plt.show()