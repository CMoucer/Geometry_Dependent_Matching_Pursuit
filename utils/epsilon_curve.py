import numpy as np

def compute_epsilon_curve(N, datas):
    """
    Given iterates produces by an optimization method for different dimensions d, this function returns an epsilon array,
    that is, given a precision level e, it returns the iteration number k(e) at which a method attains this precision.
    The precision is taken in a logarithmic scale, such that e = 1, ..., 1^-N.
    :param N: precision level
    :param datas: iterates (np.ndarray)
    :return: epsilon array (np.ndarray)
    """
    K, num_iters = datas.shape
    epsilon = [10**-(k) for k in range(N)]
    epsilon.append(0)
    ks = np.zeros((K, len(epsilon)))

    for k in range(K):
        for i in range(num_iters):
            compare = np.abs(datas[k][i]/datas[k][0])
            for e in range(N):
                if compare <= epsilon[e] and compare > epsilon[e + 1] and ks[k][e] == 0.:
                    ks[k][e] = i
                elif compare == epsilon[-1] and ks[k][-1] == 0.:
                    ks[k][-1] = i
                    for j in range(N):
                        if ks[k][j] == 0.:
                            ks[k][j] = ks[k][-1]
        for e in range(1, N+1):
            if ks[k][e] == 0.:
                ks[k][e] = 10 ** (9)
        # all initial epsilon (taken to one) are achieved at iteration 1
        ks[k][0] = 1.
    return ks, epsilon