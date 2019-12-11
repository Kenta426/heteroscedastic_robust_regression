import numpy as np
import numpy.random as npr
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.append(os.path.expanduser(".."))
from data.synthetic import generate_data_function
from model.torch_model import PolyRegression
from model.modal_lr import ModalLinearRegression
from train_loop import train_regular, train_adaptive, train_locally_adaptive
from likelihood import Gaussian, Laplace, Adaptive
from tqdm import tqdm
import gpytorch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
plt.style.use('ggplot')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def indep_noise(x):
    return npr.normal(0, 1, len(x))


def linear_noise(x):
    return npr.normal(0, np.maximum(1, 0.5*x+3), len(x))


def exp_noise(x):
    return npr.normal(0, np.exp(1/3*x), len(x))


def unimodal_noise(x):
    return npr.normal(0, 1.3/(np.maximum(np.abs(x), 0.3)), len(x))


def bimodal_noise(x):
    return npr.normal(0, np.abs(np.abs(x+2)-4)/3, len(x))


def trimodal_noise(x):
    return npr.normal(0, 1, len(x))*(np.sin((x-5))*2)


def polynomial_outcome(X):
    return 1/5*X**2


def sinusoidal_outcome(X):
    return 1 / 5 * X ** 2 + 2 * np.sin(2 * X) - 0.5 * X


def run_experiment(trX, trY, teX, teY, degree=2, degree2=2):
    x, y = torch.Tensor(trX), torch.Tensor(trY)
    tx, ty = torch.Tensor(teX), torch.Tensor(teY)
    sortedx, idxX = torch.sort(tx)
    sortedy = ty[idxX]

    adaptive = Adaptive()
    gaussian = Gaussian()
    laplace = Laplace()

    # linear regression
    stats = []
    lr = PolyRegression(degree)
    fit = train_regular(lr, x, y, gaussian, epoch=1000, learning_rate=1e-2, verbose=False)
    res1 = fit(sortedx).detach().numpy().flatten() - sortedy.numpy().flatten()
    data = dict()
    data['model'] = 'LR+' + str(degree)
    data['likelihood'] = gaussian.loglikelihood(res1)
    stats.append(data)

    lr = PolyRegression(degree)
    fit = train_regular(lr, x, y, laplace, epoch=1000, learning_rate=1e-2, verbose=False)
    res1 = fit(sortedx).detach().numpy().flatten() - sortedy.numpy().flatten()
    data = dict()
    data['model'] = 'RobustLR+' + str(degree)
    data['likelihood'] = laplace.loglikelihood(res1)
    stats.append(data)

    # adaptive linear regression
    lr = PolyRegression(degree)
    fit, alpha, scale = train_adaptive(lr, x, y, epoch=1000, learning_rate=1e-2, verbose=False)
    res = fit(sortedx).view(-1) - sortedy
    data = dict()
    data['model'] = 'Adaptive+' + str(degree)
    data['likelihood'] = adaptive.loglikelihood(res, alpha, scale)
    stats.append(data)

    # locally adaptive linear regression
    lr = PolyRegression(degree)
    alpha_model = PolyRegression(degree2, init_zeros=True)
    scale_model = PolyRegression(degree2, init_zeros=True)
    fit, alpha_reg, scale_reg = train_locally_adaptive(lr, alpha_model, scale_model, x, y,
                                                       epoch=1000, learning_rate=1e-2, verbose=False)
    res = fit(sortedx).view(-1) - sortedy
    alphas = torch.exp(alpha_reg(sortedx).view(-1))
    scales = torch.exp(scale_reg(sortedx).view(-1))

    data = dict()
    data['model'] = 'LocalAdaptive+' + str(degree)
    data['likelihood'] = adaptive.loglikelihood(res, alphas, scales)
    stats.append(data)

    # gaussian process regression
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x, y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 100
    for _ in tqdm(range(training_iter)):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(sortedx))
    # the source code divides mll by the size of input -> reconstruct mll
    data = dict()
    data['model'] = 'GPR'
    data['likelihood'] = (mll(observed_pred, sortedy) * len(sortedy)).detach().numpy()
    stats.append(data)

    return pd.DataFrame(stats)


if __name__ == '__main__':
    X = pd.read_csv('dataset/lidar.tsv', sep='  ', engine='python')
    x_range = X['range']
    y_ratio = X['logratio']
    x_range = np.array((x_range - np.mean(x_range)) / np.std(x_range))
    y_ratio = np.array((y_ratio - np.mean(y_ratio)) / np.std(y_ratio))

    x_out1 = np.random.uniform(low=-0.8, high=-0.5, size=(20,))
    y_out1 = np.random.uniform(low=-1.5, high=-1, size=(20,))
    x_out2 = np.random.uniform(low=1, high=1.5, size=(6,))
    y_out2 = np.random.uniform(low=0.5, high=1, size=(6,))

    x_range_out = np.concatenate((x_range, x_out1), axis=0)
    x_range_out = np.concatenate((x_range_out, x_out2), axis=0)
    y_ratio_out = np.concatenate((y_ratio, y_out1), axis=0)
    y_ratio_out = np.concatenate((y_ratio_out, y_out2), axis=0)

    idx = np.arange(len(x_range_out))
    np.random.shuffle(idx)
    x_range_out = x_range_out[idx]
    y_ratio_out = y_ratio_out[idx]
    kf = KFold(n_splits=5, shuffle=False)
    dfs = []

    X_train, X_test, y_train, y_test = train_test_split(x_range_out, y_ratio_out, test_size=0.2, random_state=42)
    df = run_experiment(X_train, y_train, X_test, y_test, degree=4, degree2=2)
    dfs.append(df)
    df.to_csv('results/6_5/LIDARLikelihood.csv')

    X = pd.read_csv('dataset/mcycle.csv')
    x_range = X['times']
    y_ratio = X['accel']
    x_range = np.array((x_range - np.mean(x_range)) / np.std(x_range))
    y_ratio = np.array((y_ratio - np.mean(y_ratio)) / np.std(y_ratio))

    x_out1 = np.random.uniform(low=-1.5, high=-1, size=(4,))
    y_out1 = np.random.uniform(low=-1, high=-1.4, size=(4,))
    x_out2 = np.random.uniform(low=1.3, high=1.5, size=(5,))
    y_out2 = np.random.uniform(low=-1, high=-1.5, size=(5,))

    x_range_out = np.concatenate((x_range, x_out1), axis=0)
    x_range_out = np.concatenate((x_range_out, x_out2), axis=0)
    y_ratio_out = np.concatenate((y_ratio, y_out1), axis=0)
    y_ratio_out = np.concatenate((y_ratio_out, y_out2), axis=0)

    idx = np.arange(len(x_range_out))
    np.random.shuffle(idx)
    x_range_out = x_range_out[idx]
    y_ratio_out = y_ratio_out[idx]
    X_train, X_test, y_train, y_test = train_test_split(x_range_out, y_ratio_out, test_size=0.2, random_state=42)
    df = run_experiment(X_train, y_train, X_test, y_test, degree=6, degree2=3)
    df.to_csv('results/6_5/MotorLikelihood.csv')