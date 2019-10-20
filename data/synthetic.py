"""
Synthetic data, mixture of exponential families?
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
np.random.seed(1)


def data1(n):
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


if __name__ == "__main__":
    X, y, U = data1(500)
    sns.scatterplot(X, y, hue=U)
    plt.show()
