"""
Implementation of Modal Linear Regression (Yao et al, 2014)
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


class ModalLinearRegression(object):
    def __init__(self, kernel='gaussian', bandwidth=1, maxitr=100):
        """
        :param w_dim: dimension of input covariates
        :param kernel: type of kernel
        :param bandwidth: bandwidth for kernel
        """
        self.kernel_type = kernel
        self.bandwidth = bandwidth
        self.maxitr = maxitr
        self.coef_ = None

    def kernel(self, x, y):
        assert self.kernel_type in {'gaussian', 'linear', 'exponential'}, "Invalid Kernel"
        if self.kernel_type == 'gaussian':
            return np.exp(-((y-x)/self.bandwidth)**2)
        if self.kernel_type == 'linear':
            return 0 if (y-x) > self.bandwidth else 1-(y-x)/self.bandwidth

    def fit(self, x, y):
        n, dim = x.shape
        X = np.hstack([x, np.ones(n).reshape(-1, 1)])  # insert bias
        w = np.zeros([dim+1])
        print('Fit with the EM algorithm')
        for _ in tqdm(range(self.maxitr)):
            w1 = self.E(X, y, w)
            w = self.M(X, y, w1)
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]

    def E(self, X, y, w):
        if w is None:
            return 0
        else:
            y_hat = np.dot(X, w)
            K = self.kernel(y_hat, y)
            return K/K.sum()

    def M(self, X, y, w):
        w = np.eye(len(w))*w  # sanity check passed. without this line Modal LR becomes identical to LR
        Xy = np.dot(np.dot(X.T, w), y)
        XX = np.linalg.inv(np.dot(np.dot(X.T, w), X))
        return np.dot(XX, Xy)


if __name__ == "__main__":
    X = np.random.random([3000, 2])
    beta = np.array([1, 2])
    y = np.random.normal(np.dot(X, beta), 1)
    # testing
    lr = LinearRegression()
    lr.fit(X, y)
    print(lr.coef_, lr.intercept_)
    mr = ModalLinearRegression()
    mr.fit(X, y)
    print(mr.coef_, mr.intercept_)