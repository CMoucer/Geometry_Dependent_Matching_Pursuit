import numpy as np
import matplotlib.pyplot as plt


def generate_uniform_gaussian_data(d, n, w_, sigma=1, nu=1, noise='none'):
    """
    Generate uniform Gaussian i.i.d. data
    :param d: dimension
    :param n: number of samples
    :param w_: optimal point (np.ndarray)
    :param sigma: noise parameter
    :return: labels to predict, input data and noise
    """
    ## Generate X as a normal distribution
    X = np.random.multivariate_normal(np.zeros(d), nu*np.eye(d), n)
    if noise == 'test':
        E = np.random.normal(0, sigma, d)
        y = (w_ + E) @ X.T
    else:
        E = np.random.normal(0, sigma, n)
        y = w_ @ X.T + E
    return y, X, E


def generate_non_uniform_gaussian_data(d, n, w_, sigma=1,nu=1):
    """
    Generate gaussian i.i.d data with non uniform diagonal.
    :param d: dimension
    :param n: number of samples
    :param w_ : optimal point (np.ndarray)
    :param sigma: noise parameter
    :return: labels to predict, input data and noise
    """
    ## generate points
    diag = np.eye(d)/(np.arange(d)+1)
    H = np.random.rand(d, d)
    Q,R = np.linalg.qr(H) # generate an orthogonal matrix Q
    D = Q.T @ diag @ Q

    ## Generate X as a normal distribution
    X = np.random.multivariate_normal(np.zeros(d), nu*D, n)
    E = np.random.normal(0, sigma, n)
    y = w_@X.T + E
    return y, X, E


def generate_controlled_gaussian_data(d, n, w_, sigma=0.025, omega=0.25):
    """
    We generate gaussian data such that eigenvalues of sigma are in [1/(1 + omega)^2, 2/(1-omega)^2/(1+omega)]
    :param d: dimension
    :param n: number of samples
    :param w_: optimal point (np.ndarray)
    :param sigma: noise parameter
    :param omega: range parameter
    :return: labels to predict, input data and noise
    """
    X = np.zeros((n, d))
    for i in range(n):
        z = np.random.normal(0, 1, d)
        X[i][0] = z[0]/(1 - omega**2)
        for j in range(d-1):
            X[i][j+1] = X[i][j] * omega + z[j+1]
    E = np.random.normal(0, sigma, n)
    y = w_ @ X.T + E
    return y, X, E

if __name__=='__main__':
    n = 50
    d = 500
    trials = 200
    Ls = np.zeros(trials)
    print(np.log(d)/n)
    for k in range(trials):
        w_ = np.random.rand(d)
        _, X, _ = generate_uniform_gaussian_data(d=d, n=n, w_=w_, sigma=1)
        Ls[k] = max([np.linalg.norm(X[:, i]) ** 2 for i in range(d)]) / n

    plt.hist(Ls, bins=10)
    plt.show()

