"""
Regression with General Adaptive Loss (Barron et al, 2019)
"""
import torch
import robust_loss_pytorch.general
import numpy as np
from tqdm import tqdm


class RegressionModel(torch.nn.Module):
    # A simple linear regression module.
    def __init__(self, dim):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x[:, None])[:, 0]


class Poly2RegressionModel(torch.nn.Module):
    # A simple linear regression module.
    def __init__(self, dim):
        super(Poly2RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(dim*2, 1)

    def forward(self, x):
        x = torch.cat([(x ** 2).view(-1, 1), x.view(-1, 1)], axis=1)
        return self.linear(x)[:, 0]


class AdaptiveRegression(object):
    def __init__(self, base_model='linear', epoch=1000, lr=0.01):
        assert base_model in {'linear', 'polynomial'}, "Invalid base model"
        self.base_model = base_model
        self.epoch = epoch
        self.coef_ = None
        self.intercept_ = None
        self.alpha = None
        self.scale = None
        self.base = None
        self.lr = lr

    def fit(self, X, y):
        n, dim = X.shape
        if self.base_model == 'linear':
            base = RegressionModel(dim)
        if self.base_model == 'polynomial':
            base = Poly2RegressionModel(dim)
        adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=1, float_dtype=np.float32, device='cpu')
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        params = list(base.parameters()) + list(adaptive.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for _ in tqdm(range(self.epoch)):
            y_i = base(X).squeeze()
            # Stealthily unsqueeze to an (n,1) matrix, and then compute the loss.
            # A matrix with this shape corresponds to a loss where there's one shape+scale parameter
            # per dimension (and there's only one dimension for this data).
            loss = torch.mean(adaptive.lossfun((y_i - y)[:, None]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.coef_ = list(base.parameters())[0].data.numpy()[0]
        self.intercept_ = list(base.parameters())[1].data.numpy()[0]
        self.alpha = adaptive.alpha()[0, 0].data.numpy()
        self.scale = adaptive.scale()[0,0].data.numpy()
        self.base = base

    def predict(self, X):
        X = torch.Tensor(X)
        return self.base(X).data.numpy().flatten()


if __name__ == "__main__":
    X = np.random.random([3000, 2])
    beta = np.array([1, 2])
    y = np.random.normal(np.dot(X, beta), 1)
    # testing
    lr = AdaptiveRegression()
    lr.fit(X, y)
    lr.predict(X)
    print(lr.coef_, lr.intercept_)
    print(lr.alpha, lr.scale)