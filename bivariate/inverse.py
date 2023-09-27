# Functions for working with bivariate denoisor prior
import torch, numpy as np
import torch.nn as nn
import torch.nn.utils.parametrizations as para
from torch.linalg import vector_norm as vnorm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
N_DIM = 2

# sampling function for bivariate prior
# 2-D special case for the algorithm in inverse/sampler.py
def sample(model, device, h=0.20, beta=0.20, end=0.025, min_t=4):
    # setup initial values
    init = torch.rand(N_DIM) * 2 - 1
    y = init.unsqueeze(0).to(device)
    sigma = torch.norm(model.score(y)) / np.sqrt(N_DIM)
    scale = np.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2)

    # record all steps
    t = 1
    all_y = []
    all_y.append(init.numpy())

    # iterative denoising
    while sigma > end:
        # compute gradient
        d = model.score(y)
        sigma = torch.norm(d) / np.sqrt(N_DIM)

        # injected noise
        noise = torch.randn(size=y.size(), device=device)

        # update sample
        y = y + h * d + scale * sigma * noise

        # record
        t = t + 1
        all_y.append(y.detach().squeeze().cpu().numpy())

    # set threshold for minimum number of steps
    if t < min_t:
        return sample(model, device, h, beta, end, min_t)

    # return the sample trajectory
    return np.array(all_y)

# linear inverse function for bivariate prior and 1-D projection
# 2-D special case for the algorithm in inverse/orthogonal.py
class LinearInverse(nn.Module):
    N_SAMPLE = 1
    N_DIM = 2

    def __init__(self, model):
        super().__init__()

        # save variables
        self.model = model

        # no grad flag for the denoiser model
        for param in self.model.parameters():
            param.requires_grad = False

        # 1-D measurement vector
        linear = nn.Linear(self.N_DIM, self.N_SAMPLE)
        self.linear = para.orthogonal(linear, orthogonal_map='householder')
        # row vector of (N_SAMPLE = 1, N_DIM = 2)
        self.vector = self.linear.weight

        # parameters for the reconstruction
        self.h = 0.20
        self.beta = 0.20
        self.end = 0.025
        self.last_t = None

    def refresh(self):
        self.vector = self.linear.weight

    def assign(self, vector):
        self.linear.weight = vector
        self.refresh()

    def to(self, device):
        return_val = super().to(device)
        self.refresh()
        return return_val

    def log_grad(self, x):
        # return log grad p(y)
        return self.model.score(x)

    def measure(self, x):
        return torch.matmul(x, self.vector.t())

    def recon(self, m):
        return torch.matmul(m, self.vector)

    def inverse(self, m):
        # measurement vector calculation
        M = self.measure
        M_T = self.recon

        # init variables
        proj = M_T(m)
        n = torch.numel(proj[0])
        y = torch.randn_like(proj) + proj
        scale = np.sqrt((1 - self.beta * self.h) ** 2
                        - (1 - self.h) ** 2)

        # corse-to-fine sampling
        t = 1
        flag = True
        while flag:
            # projected log prior gradient
            d = self.log_grad(y)
            d = (d - M_T(M(d)) + proj - M_T(M(y)))

            # compute noise magnitude for stopping criterion
            sigma = vnorm(d, dim=(1)) / np.sqrt(n)
            if torch.max(sigma) <= self.end:
                flag = False

            # injected noise
            noise = torch.randn_like(y)
            gamma = scale * sigma
            # expand gamma for shape matching
            gamma = gamma[:, None].repeat([1, self.N_DIM])

            # update image
            y = y + self.h * d + gamma * noise
            t += 1

        # save the number of steps
        self.last_t = t
        return y

    def forward(self, x):
        self.refresh()

        m = self.measure(x)
        return self.inverse(m)

    # reconstruction using sample average
    def average(self, x, sample=2):
        # make copies of the input
        x = x.repeat([sample, 1])

        # run the inverse algorithm
        recon = self.forward(x)

        # average the results
        recon = recon.reshape([sample, -1, self.N_DIM])
        recon = torch.mean(recon, dim=0)

        return recon

def lnopt(solver, train, test, device, batch_size=128,
          n_epoch=50, lr=1e-3, gamma=0.95, verbose=True):

    # training and test data
    train_data = DataLoader(train, batch_size, shuffle=True,
                            num_workers=8, pin_memory=True)
    test = test.to(device)

    # optimizers
    loss = nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(solver.parameters(), lr=lr)
    scheduler = ExponentialLR(optim, gamma=gamma)

    batch_loss = []
    epoch_loss = []

    # Run n_epoch of training
    for epoch in range(n_epoch):
        total_loss = 0.0

        # SGD Optimization
        for count, batch in enumerate(train_data):
            optim.zero_grad(set_to_none=True)

            batch = batch.to(device)
            recon = solver.average(batch, sample=2)
            error = loss(recon, batch)

            # optim step
            error.backward()
            optim.step()

            # record loss value
            loss_val = error.item() / batch.shape[0]
            batch_loss.append(loss_val)
            total_loss += loss_val

        # average loss value per batch
        avg_loss = total_loss / float(count)
        epoch_loss.append(avg_loss)

        # adjust learning rate
        scheduler.step()

        # compute performance on test set
        with torch.no_grad():
            recon_test = solver.average(test, sample=2)
            test_loss = loss(recon_test, test).item() / test.shape[0]

        # print training information
        if verbose:
            print('Epoch %d/%d' % (epoch + 1, n_epoch))
            print('Training loss %.3f' % (avg_loss))
            print('Test loss %.3f \n' % test_loss)

    return np.array(batch_loss), np.array(epoch_loss)