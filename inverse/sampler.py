import torch, numpy as np

# sample from the implicit prior
def sample_prior(model, init, h_init=0.01, beta=0.01, sig_end=0.01, stride=5):
    '''
    h: step size of the gradient step
    beta: amount of injected noise
    sig_end: stop criterion
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)

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
        # update step size
        h = (h_init * t) / (1 + h_init * (t - 1))

        # log prior gradient and estimate noise magnitude
        d = log_grad(y)
        sigma = torch.norm(d) / np.sqrt(n)

        # amount of injected noise
        gamma = np.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2) * sigma
        noise = torch.randn(size=y.size(), device=device)
        
        # update image
        y = y + h * d + gamma * noise

        if stride > 0 and (t - 1) % stride == 0:
            print('iter %d, sigma %.2f' % (t, sigma.item()))
            all_ys.append(y.squeeze(0).permute(1, 2, 0).cpu().numpy())

        t += 1

    # different convention for numpy vs pytorch images
    all_ys.append(y.squeeze(0).permute(1, 2, 0).cpu().numpy())

    return all_ys