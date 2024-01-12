"""
This file returns smoothness and strong-convexity parameters for a linear regression model f(u) = 1/n \|Xu - y\|_2^2
built on :
- synthetic i.i.d. gaussian data,
- a random feature models obtained from the Leukemia dataset.
More precisely it computes the exact smoothness parameter with respect to the ||.||_2 and ||.||_1-norm, the exact
strong convexity and Lojasiewicz parameter with respect to the ||.||_2 norm, and SDP relaxations for strong convexity
and Lojasiewicz parameter with respect to the ||.||_1-norm.
"""

import numpy as np

from generate_data.compute_mu import compute_mu1_underparametrized, compute_mu1_overparametrized
from generate_data.gaussian import generate_uniform_gaussian_data
from generate_data.sparse_optimum import generate_sparse_optimum
from generate_data.real_datasets import load_leukemia, preprocess_standard_scale
from generate_data.generate_random_features import return_RF
np.random.seed(0)


### PARAMETERS
#Models may be 'gaussian', 'leukemia' or 'rf'
model = 'rf'

# Generate the random model
if model == 'gaussian':
    n = 10
    d = 20
    s = 4
    sigma = 0.5
    w_ = generate_sparse_optimum(d=d, s=s)
    y, X, E = generate_uniform_gaussian_data(d=d, n=n, w_=w_, sigma=sigma)

elif model == 'rf':
    n = 10 # number fo samples from the Leukemia dataset
    d = 20 # dimension of the output of the random feature model
    X_leuk, y_leuk = load_leukemia()
    sigma = 0.5  # variance of theta
    m_r = 50 # dimension of the features
    theta = np.random.multivariate_normal(np.zeros(d), sigma * np.eye(d), size=m_r)
    X_r = preprocess_standard_scale(X_leuk[:n, :], d)
    X = return_RF(X_r, theta, d)

## Compute smoothness, strong-convexity and Lojqsiewicz constants
print('__SMOTHNESS__')
L_2 = max(np.linalg.eigvalsh(X.T @ X)) / n
print('Smoothness with respect to the ||.||_2 norm', L_2)
L_1 = max([np.linalg.norm(X[:, j]) ** 2 for j in range(d)]) / n
print('Smoothness with respect to the ||.||_1 norm', L_1)
print(' ')
print('__STRONG CONVEXITY and LOJASIEWICZ CONSTANT__')
if d > n:
    print ('Overparametrized regime')
    mu_1L = compute_mu1_overparametrized(X)/n
    mu_1 = 0
    print('Strong convexity with respect to the ||.||_1 norm', mu_1)
    print('Lojasiewicz inequality with respect to the ||.||_1 norm', mu_1L)
else:
    print('Underparametrized regime')
    mu_1L = 0
    mu_1 =  compute_mu1_underparametrized(X)/n
    print('Strong convexity with respect to the ||.||_1 norm', mu_1)
    print('Lojasiewicz inequality with respect to the ||.||_1 norm', mu_1L)
mu_2 = min(np.linalg.eigvalsh((X.T @ X) / n))
print('Strong convexity with respect to the ||.||_2 norm', mu_2)
mu_2L = min(np.linalg.eigvalsh(X @ X.T)) / n
print('Lojasiewicz inequality with respect to the ||.||_2 norm', mu_2L)
