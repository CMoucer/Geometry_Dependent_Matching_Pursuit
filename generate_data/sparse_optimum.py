import numpy as np

def generate_sparse_optimum(d, s):
    """
    This method returns a sparse vector with sparsity s.
    :param d: dimension
    :param s: sparsity
    :return:
    """
    w_ = 2*np.random.randint(2, size=d)-1
    I = np.random.randint(0, d, s)
    N = np.arange(d)
    N_droped = np.delete(N, I)
    w_[N_droped] = 0  # create a sparse vector
    return w_

def generate_sparse_simplex(d, s, R=1.2):
    """
    This method returns a sparse vector adapted to the simplex with sparsity s.
    :param d: dimension
    :param s: sparsity
    :param R: extension to be slightly outside of the simplex.
    :return:
    """
    w_ = np.random.random(d)
    I = np.random.randint(0, d, s)
    N = np.arange(d)
    N_droped = np.delete(N, I)
    w_[N_droped] = 0  # create a sparse vector
    w_ /= np.linalg.norm(w_, 1)
    w_ *=R
    return w_

if __name__=='__main__':
    d = 200
    s = 8
    w_ = generate_sparse_optimum(d, s)
    w_simplex = generate_sparse_simplex(d, s)
    print(w_simplex)
    print(np.sum(w_simplex))