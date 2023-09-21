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

