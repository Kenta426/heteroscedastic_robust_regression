"""
Synthetic data, mixture of exponential families?
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
np.random.seed(1)


def data1(n):
    """
    Initial idea of mixture of two distribution, one with linear and another with polynomial
    :param n:
    :return:
    """
    scale_true = 0.7
    shift_true = 0.15
    scale2_true = 2
    p = 0.5  # rate to be a member of one group
    U = np.random.binomial(1, p, n)
    x = np.random.uniform(size=n)
    y = scale_true * x + shift_true
    y[U == 1] = scale2_true * x[U == 1] ** 2 + shift_true
    y += np.random.normal(scale=0.025, size=n)
    flip_mask = np.random.uniform(size=n) > 0.9
    y = np.where(flip_mask, 0.05 + 0.4 * (1. - np.sign(y - 0.5)), y)
    return x, y, U


def data2():
    """
    Data from ML HW
    """
    train_X = np.load('data/q3_train_X.npy')
    train_y = np.load('data/q3_train_y.npy')
    return train_X, train_y


def data3(n):
    """
    data with heteroskedastic noise
    """
    p = 0.3
    U = np.random.binomial(1, p, n)
    X = np.zeros(n)
    X[U == 1] = np.random.normal(1, 4, sum(U))
    X[U == 0] = np.random.normal(0, 1, n-sum(U))
    Y = 2*X**2 - 2*X + 0.1 + np.random.normal(0, 1, n)*X**2
    return X, Y, U


def generate_delta_noise(n: int, x_range: list, delta=10, rate=0.01, beta1=3, beta0=0, scale=1, hetero=False):
    X = npr.uniform(x_range[0], x_range[1], n)
    X = np.array(sorted(X))
    scales = scale * np.ones(len(X))
    if hetero:
        scales[X > 0] += (X ** 2 + 5 * X)[X > 0]
    Y = npr.normal(beta1 * X + beta0, scales)

    noise = np.array([False] * n)
    idx = np.random.choice(int(n / 5), int(n * rate), replace=False)
    noise[idx] = True
    Y[noise] = npr.normal(beta1 * (X[noise] + delta) + beta0)
    return X, Y, noise


def generate_data_function(base_func, noise_func, n, test_data = 0.2, rate=0.01, loc=[2,5], yloc=[15, 15]):
    """
    n : number of data to generate
    rate : rate of outliers (set 0 if you don't want outlier)
    loc : mean location on x-axis for outliers
    y : mean for outliers

    return
    X : x-coordinate
    output : y-coordinate from noise less model
    Y : noisy output
    """
    # base data
    X = npr.uniform(-5, 5, n)
    output = base_func(X)
    noise = noise_func(X)
    Y = output+noise

    trainX = X[:int(n*(1-test_data))]
    testX = X[int(n*(1-test_data)):]
    trainY = Y[:int(n*(1-test_data))]
    testY = Y[int(n*(1-test_data)):]

    # generate outliers
    if not isinstance(loc, list):
        loc = [loc]
    if not isinstance(yloc, list):
        yloc = [yloc]
    assert len(yloc) == len(loc), 'location of outliers must match'
    out = int(n*(1-test_data)*rate/len(loc))
    Xout = []
    Yout = []
    for l, yl in zip(loc, yloc):
        out_data = npr.multivariate_normal([0,0], [[1,0],[0,1]], out)
        out_x = out_data[:, 0]+l
        out_y = out_data[:, 1]+yl
        Xout.append(out_x)
        Yout.append(out_y)
    Xout = np.hstack(Xout)
    Yout = np.hstack(Yout)
    trainX = np.hstack([trainX, Xout])
    trainY = np.hstack([trainY, Yout])

    trainX = (trainX-trainX.mean())/trainX.std()
    trainY = (trainY-trainY.mean())/trainY.std()
    testX = (testX-testX.mean())/testX.std()
    testY = (testY-testY.mean())/testY.std()

    return trainX, trainY, testX, testY


if __name__ == "__main__":
    X, y, U = data3(500)
    sns.scatterplot(X, y, hue=U)
    plt.show()
    # data2()

