# Functions for working with bivariate denoisor prior
import torch, numpy as np
import torch.nn as nn
import torch.nn.utils.parametrizations as para
from torch.linalg import vector_norm as vnorm
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
        self.max_t = 200
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
        e = torch.ones_like(proj)
        n = torch.numel(e[0])
        mu = 0.5 * (e - M_T(M(e))) + proj
        y = torch.randn_like(mu) + mu
        scale = np.sqrt((1 - self.beta * self.h) ** 2 - (1 - self.h) ** 2)
        sigma = vnorm(self.log_grad(y), dim=(1)) / np.sqrt(n)

        # corse-to-fine sampling
        t = 1
        while torch.max(sigma) > self.end:
            # projected log prior gradient
            d = self.log_grad(y)
            d = (d - M_T(M(d)) + proj - M_T(M(y)))

            # compute noise magnitude
            sigma = vnorm(d, dim=(1)) / np.sqrt(n)

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

        # run a final denoise step and return the results
        return y + self.log_grad(y)

    def forward(self, x):
        self.refresh()

        m = self.measure(x)
        return self.inverse(m)