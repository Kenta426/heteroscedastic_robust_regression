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
    return npr.normal(0, 0.5/(np.maximum(np.abs(x), 0.3)), len(x))


def bimodal_noise(x):
    return npr.normal(0, np.abs(np.abs(x+2)-4)/3, len(x))


def trimodal_noise(x):
    return npr.normal(0, 1, len(x))*(np.sin((x-5))*2)


def polynomial_outcome(X):
    return 1/5*X**2


def sinusoidal_outcome(X):
    return 1 / 5 * X ** 2 + 2 * np.sin(2 * X) - 0.5 * X


def run_experiment(trX,trY, teX, teY, degree=2):
    x, y = torch.Tensor(trX), torch.Tensor(trY)
    tx, ty = torch.Tensor(teX), torch.Tensor(teY)
    sortedx, idxX = torch.sort(tx)
    sortedy = ty[idxX]

    gaussian = Gaussian()
    laplace = Laplace()
    adaptive = Adaptive()
    # linear regression
    stats = []

    lr = PolyRegression(degree)
    fit = train_regular(lr, x, y, gaussian, epoch=100, learning_rate=1e-2, verbose=False)
    data = dict()
    data['model'] = 'LR+' + str(degree)
    param = fit.beta.weight.data.numpy().flatten()
    bias = fit.beta.bias.data.numpy().flatten()
    data['param0'] = param[0]
    data['param1'] = param[1]
    data['param2'] = bias[0]

    stats.append(data)

    # robust linear regression
    lr = PolyRegression(degree)
    fit = train_regular(lr, x, y, laplace, epoch=100, learning_rate=1e-2, verbose=False)
    data = dict()
    data['model'] = 'RobustLR+' + str(degree)
    param = fit.beta.weight.data.numpy().flatten()
    bias = fit.beta.bias.data.numpy().flatten()
    data['param0'] = param[0]
    data['param1'] = param[1]
    data['param2'] = bias[0]

    stats.append(data)

    # adaptive linear regression
    lr = PolyRegression(degree)
    fit, alpha, scale = train_adaptive(lr, x, y, epoch=100, learning_rate=1e-2, verbose=False)
    data = dict()
    data['model'] = 'Adaptive+' + str(degree)
    param = fit.beta.weight.data.numpy().flatten()
    bias = fit.beta.bias.data.numpy().flatten()
    data['param0'] = param[0]
    data['param1'] = param[1]
    data['param2'] = bias[0]
    stats.append(data)

    # locally adaptive linear regression
    lr = PolyRegression(degree)
    alpha_model = PolyRegression(2, init_zeros=True)
    scale_model = PolyRegression(2, init_zeros=True)
    fit, alpha_reg, scale_reg = train_locally_adaptive(lr, alpha_model, scale_model, x, y,
                                                       epoch=500, learning_rate=1e-2, verbose=False)
    data = dict()
    data['model'] = 'LocalAdaptive+' + str(degree)
    param = fit.beta.weight.data.numpy().flatten()
    bias = fit.beta.bias.data.numpy().flatten()
    data['param0'] = param[0]
    data['param1'] = param[1]
    data['param2'] = bias[0]
    stats.append(data)

    # modal regression
    ml = ModalLinearRegression(kernel="gaussian", poly=degree, bandwidth=1)
    ml.fit(x.numpy().reshape(len(x), -1), y.numpy().reshape(-1))
    yml = ml.predict(sortedx.detach().numpy().reshape(len(sortedx), -1))
    data = dict()
    data['model'] = 'Modal'
    param = ml.coef_.flatten()
    bias = ml.intercept_.flatten()
    data['param0'] = param[0]
    data['param1'] = param[1]
    data['param2'] = bias[0]
    stats.append(data)

    return pd.DataFrame(stats)


if __name__ == '__main__':
    n = 1000
    output_func = polynomial_outcome
    noise_func = indep_noise
    n_name = ['Tri']
    # n_name = ['Indep', 'Linear', 'Exp', 'Uni', 'Bi', 'Tri']
    noise = [indep_noise, linear_noise, exp_noise, unimodal_noise, bimodal_noise, trimodal_noise]
    Y_name = ['Poly']
    output = [polynomial_outcome, sinusoidal_outcome]
    for i in range(len(Y_name)):
        for j in range(len(n_name)):
            dfs = []
            for k in range(5):
                trX, trY, teX, teY = generate_data_function(output[i], noise[j], n, rate=0.1, loc=[-2], yloc=[10])
                if i == 0:
                    degree = 2
                else:
                    degree = 5
                df = run_experiment(trX, trY, teX, teY, degree)
                df['rep'] = k
                dfs.append(df)
            pd.concat(dfs).to_csv('results/6_2_2/'+Y_name[i]+n_name[j]+'Outlier.csv')
