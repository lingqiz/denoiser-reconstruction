import torch, numpy as np
import warnings

# sample from the implicit prior
def sample_prior(model, init, h_init=0.01, beta=0.01, 
                 sig_end=0.005, stride=10, fix_h=False):
    '''
    h: step size of the gradient step
    beta: amount of injected noise
    sig_end: stop criterion
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)

    # helper function for pytorch image to numpy image
    numpy_image = lambda y: y.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # the network calculates the noise residual
    def log_grad(y):
        with torch.no_grad():
            return - model(y)

    # init variables
    n = torch.numel(init)
    y = init.unsqueeze(0).to(device)
    sigma = torch.norm(log_grad(y)) / np.sqrt(n)

    t = 1
    all_ys = []
    while sigma > sig_end:
        # determine step size
        if fix_h:
            h = h_init
        else:            
            h = (h_init * t) / (1 + h_init * (t - 1))

        # log prior gradient and estimate noise magnitude
        d = log_grad(y)
        sigma = torch.norm(d) / np.sqrt(n)

        # protect against divergence
        div_thld = 1e2
        if sigma > div_thld:
            warnings.warn('Divergence detected, resample with \
                larger step size and tolerance.', RuntimeWarning)

            return sample_prior(model, init,
            h_init, beta * 2, sig_end * 2, stride)

        # inject noise
        gamma = np.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2) * sigma
        noise = torch.randn(size=y.size(), device=device)

        # update image
        y = y + h * d + gamma * noise

        if stride > 0 and (t - 1) % stride == 0:
            print('iter %d, sigma %.2f' % (t, sigma.item()))
            all_ys.append(numpy_image(y))

        t += 1

    all_ys.append(numpy_image(y))
    return all_ys