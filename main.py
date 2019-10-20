"""
1. generate or load data
2. train linear regression with L2 loss
3. train linear regression with adaptive loss
4. train modal linear regression
5. check bias/variance trade-off
6. (optional) plot lines
"""
from data.synthetic import data1
from model.modal_lr import ModalLinearRegression
from model.adaptive_lr import AdaptiveRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
from prettytable import PrettyTable
plt.style.use('ggplot')


def main():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    # step 1
    n = cfg['data']['n']
    x, y, u = data1(n=n)
    x = x.reshape(n, -1)

    # step 2
    lr = LinearRegression()
    lr.fit(x, y)

    # step 3
    ml = ModalLinearRegression(bandwidth=0.3, maxitr=500)  # seems very important to calibrate bandwidth
    ml.fit(x, y)

    # step 4
    ada_lr = AdaptiveRegression()
    ada_lr.fit(x, y)

    # step 5
    table = PrettyTable()
    table.field_names = ["Model", "MSE"]
    mse_lr = np.mean((y.flatten() - lr.predict(x))**2)
    mse_lr = int(mse_lr * 1000) / 1000
    mse_ml = np.mean((y.flatten() - ml.predict(x))**2)
    mse_ml = int(mse_ml * 1000) / 1000
    mse_alr = np.mean((y.flatten() - ada_lr.predict(x))**2)
    mse_alr = int(mse_alr * 1000) / 1000

    table.add_row(["Linear Regression", mse_lr])
    table.add_row(["Modal Regression", mse_ml])
    table.add_row(["Adaptive Regression", mse_alr])
    print(table)

    # step 6
    if cfg['plot']:
        sns.scatterplot(x.flatten(), y, hue=u)
        x_plot = np.linspace(0, 1, 100)  # hacky (I know the range of X is [0,1])
        y_plot_ml = ml.predict(x_plot.reshape(100, -1))
        y_plot_alr = ada_lr.predict(x_plot.reshape(100, -1))
        y_plot_lr = lr.predict(x_plot.reshape(100, -1))
        plt.plot(x_plot, y_plot_lr, color='coral', label='Linear Regression')
        plt.plot(x_plot, y_plot_ml, color='navy', label='Modal Regression')
        plt.plot(x_plot, y_plot_alr, color='darkgreen', label='Adaptive Regression')
        plt.legend()
        # plt.savefig('fig/comparison.png')
        plt.show()

if __name__ == '__main__':
    main()