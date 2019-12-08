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
from train_loop import train_regular, train_adaptive, train_locally_adaptive
from likelihood import Gaussian, Laplace, Adaptive
plt.style.use('ggplot')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    def indep_noise(x):
        return npr.normal(0, 0.5, len(x))

    def linear_noise(x):
        return npr.normal(0, np.maximum(1,x), len(x))

    def exp_noise(x):
        return npr.normal(0, np.exp(1/3*x), len(x))

    def output_func(x):
        return 1 / 5 * x ** 2

    trX, trY, teX, teY = generate_data_function(output_func, exp_noise, 500, rate=0.1, loc=[-2], yloc=[10])

    x, y = torch.Tensor(trX), torch.Tensor(trY)
    gaussian = Gaussian()
    laplace = Laplace()
    adaptive = Adaptive()

    # # 2-D linear regression
    lr2 = PolyRegression(2)
    fit2 = train_regular(lr2, x, y, gaussian, epoch=1000, learning_rate=1e-2, verbose=False)

    # # 2-D linear regression + laplace
    robust_lr2 = PolyRegression(2)
    robust_fit2 = train_regular(robust_lr2, x, y, laplace, epoch=1000, learning_rate=1e-2, verbose=False)

    # 2-D linear regression + adaptive
    ada_lr2 = PolyRegression(2)
    ada_fit2, alpha, scale = train_adaptive(ada_lr2, x, y, epoch=1000, learning_rate=1e-2, verbose=False)

    ada_lr2 = PolyRegression(2)
    alpha_model = PolyRegression(1, init_zeros=True)
    scale_model = PolyRegression(1, init_zeros=True)

    ada_fit22, alpha_reg, scale_reg = train_locally_adaptive(ada_lr2, alpha_model, scale_model, x, y,
                                                             epoch=1000, learning_rate=1e-2, verbose=False)

    sortedx, idxX = torch.sort(x)
    y2 = fit2(sortedx).detach().numpy()
    yr2 = robust_fit2(sortedx).detach().numpy()
    yar2 = ada_fit2(sortedx).view(-1)
    yar22 = ada_fit22(sortedx).view(-1)
    alphas = torch.exp(alpha_reg(sortedx).view(-1))
    scales = torch.exp(scale_reg(sortedx).view(-1))

    # eps = y2.flatten() - sortedx.numpy().flatten()
    print('Gaussian Log Likelihood: ', gaussian.loglikelihood(y2.flatten()-y[idxX].numpy().flatten()))
    print('Laplace Log Likelihood:  ', laplace.loglikelihood(yr2.flatten()-y[idxX].numpy().flatten()))
    # # sanity check
    # # print(adaptive.loglikelihood(torch.Tensor(eps), torch.Tensor([2.0]), torch.Tensor(eps**2).mean().sqrt()))
    print('Adaptive Log Likelihood: ', adaptive.loglikelihood(yar2-y[idxX], alpha, scale))
    print('Ada+reg Log Likelihood : ', adaptive.loglikelihood(yar22-y[idxX], alphas, scales))

    #
    # # sns.lineplot(sortedx, y1.flatten(), label='lr 1d')
    sns.lineplot(sortedx, output_func(sortedx).flatten(), label='ground truth')
    sns.scatterplot(trX, trY, alpha=0.2)
    sns.lineplot(sortedx.detach(), y2.flatten(), label='lr 2d')
    # # sns.lineplot(sortedx, y3.flatten(), label='lr 3d')
    # sns.lineplot(sortedx.detach(), yr2.flatten(), label='robust 2d')
    sns.lineplot(sortedx.detach(), yar2.detach().flatten(), label='adaptive robust 2d')
    sns.lineplot(sortedx.detach(), yar22.detach().flatten(), label='adaptive robust 2d + regression')
    plt.show()