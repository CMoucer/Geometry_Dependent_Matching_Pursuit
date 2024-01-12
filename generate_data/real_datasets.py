import numpy as np

from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn import preprocessing


def diabetes():
    """
    This is a loader for diabetes data from scikit-learn.
    :return:
    """
    X, y = datasets.load_diabetes(return_X_y=True)
    assert X.shape[0] == y.shape[0], 'dimension error'
    # Preprocessing
    X /= X.std(axis=0)
    return X, y

def breast_cancer():
    """
    This is a loader for breast cancer data from scikit-learn.
    :return:
    """
    X, y = datasets.load_breast_cancer(return_X_y=True)
    assert X.shape[0] == y.shape[0], 'dimension error'
    # Preprocessing
    X /= X.std(axis=0)
    return X, y


def load_leukemia():
    """
    This is a loader for the Leukemia dataset that performs a first preprocessing step.
    :return:
    """
    print("Loading data...")
    dataset = fetch_openml("leukemia", as_frame=False, parser='liac-arff')

    X = np.asfortranarray(dataset.data.astype(float)) # features
    y = 2 * ((dataset.target != "AML") - 0.5)
    n, d = X.shape # training size
    print('sample size n=', n)
    print('dimension d=', d)

    # add random noise
    sigma = 0.1
    y += np.random.normal(loc=0.0, scale=sigma, size=n)
    y -= np.mean(y)
    y /= np.std(y)
    return X, y

def preprocess_standard_scale(X, d_r):
    """
    Reduce dimension on the leukemia dataset and apply a standard scaler preprocessing.
    :param X: input data (np.ndarray)
    :param d_r: chosen dimension
    :return:
    """
    print('chosen dimension d_r=', d_r)
    X_r = X[:, :d_r]
    scale = preprocessing.StandardScaler().fit(X_r)
    X_r = scale.transform(X_r)
    return X_r


if __name__ == '__main__':
    X, y = diabetes()
    n, d = X.shape
    print('diabetes:', n, d)
    print(np.diagonal((X.T @ X) / n))
    print(np.cov(X))

    Xb, yb = breast_cancer()
    nb, db = Xb.shape
    print('breast cancer:', nb, db)
    print(np.diagonal((Xb.T @ Xb) / n))
    print(np.cov(Xb))