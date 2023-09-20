# Functions for working with bivariate denoisor prior
import torch, numpy as np
DIM = 2

# sampling function for bivariate prior
def sample(model, device, h=0.1, beta=0.1, end=0.01):
    # setup initial values
    init = torch.rand(DIM) * 2 - 1
    y = init.unsqueeze(0).to(device)
    sigma = torch.norm(model.score(y)) / np.sqrt(DIM)
    scale = np.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2)

    # record all steps
    t = 1
    all_y = []
    all_y.append(init.numpy())

    # iterative denoising
    while sigma > end:
        # compute gradient
        d = model.score(y)
        sigma = torch.norm(d) / np.sqrt(DIM)

        # injected noise
        noise = torch.randn(size=y.size(), device=device)

        # update sample
        y = y + h * d + scale * sigma * noise

        # record
        t = t + 1
        all_y.append(y.detach().squeeze().cpu().numpy())

    return np.array(all_y)
