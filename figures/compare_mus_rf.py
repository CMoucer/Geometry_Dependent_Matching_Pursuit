"""
This file compares the values of :
- mu_1 and mu_2 in the underparametrized regime,
- mu_1^L and mu_2^L in the overparametrized regime,
for a random feature model obtained from the Leukemia dataset
"""

### IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

from generate_data.compute_mu import compute_mu1_underparametrized, compute_mu1_overparametrized
from generate_data.real_datasets import load_leukemia, preprocess_standard_scale
from generate_data.generate_random_features import return_RF


### PARAMETERS
avg = 1
n_dim = 20

## Select the dimensions and number of samples in the underparametrized and the overparametrized regime
# underparametrized regime
n_under = 50
ds_under = np.linspace(5, n_under, n_dim).astype(int)
# overparametrized regime
n_over = 20
ds_over = np.linspace(n_over, n_over * 10, n_dim).astype(int)

# Load the leukemia dataset
X_leuk, y_leuk = load_leukemia()
# random feature model
d_r = 100 # chosen the dimension for the random feature model
sigma = 0.5 # noiser
theta = np.random.multivariate_normal(np.zeros(d_r), sigma * np.eye(d_r), size=200)

mu2s = np.zeros((2, n_dim, avg))
mu1s = np.zeros((2, n_dim, avg))

for j in range(2):
    if j == 0:
        status = 'under'
        n = n_under
        ds = ds_under
    else:
        status = 'over'
        n = n_over
        ds = ds_over
    for i in range(n_dim):
        for k in range(avg):
            X_r = preprocess_standard_scale(X_leuk[:n, :], d_r)
            X = return_RF(X_r, theta, ds[i])
            if n >= ds[i]:
                # underparametrized
                mu1s[j][i][k] = compute_mu1_underparametrized(X)
                mu2s[j][i][k] = min(np.linalg.eigvalsh(X.T @ X))
            else:
                # overparametrized
                mu1s[j][i][k] = compute_mu1_overparametrized(X, n_max=31)
                mu2s[j][i][k] = min(np.linalg.eigvalsh(X @ X.T))

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

for j in range(2):
    if j == 0:
        status = 'under'
        n = n_under
        ds = ds_under
    else:
        status = 'over'
        n = n_over
        ds = ds_over
    axs[j].plot(ds, np.mean(mu1s[j, :, :], axis=1), color='red', label='SDP')
    axs[j].plot(ds, np.mean(mu2s[j, :, :], axis=1), '--', color='blue', label='mu2')
    axs[j].plot(ds, np.mean(mu2s[j, :, :], axis=1)/ds, '--', color='green', label='mu2/d')

    axs[j].set_xlabel('dimension', fontsize=16)
    if j == 0:
        axs[j].set_ylabel('mu', fontsize=16)
    else:
        axs[j].set_ylabel('mu_PL', fontsize=16)
    axs[j].set_yscale('log')
    axs[j].set_title(f'random features with n={n}', fontsize=18)
    axs[j].tick_params(axis='both', labelsize=18)
fig.suptitle(f'Comparison of SDP relaxations for mu with different variances for real data', fontsize=18)

plt.show()