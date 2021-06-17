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

    t = 1
    all_ys = []
    while sigma > sig_end:
        h = (h_init * t) / (1 + h_init * (t - 1))

        d = log_grad(y)
        sigma = torch.norm(d) / torch.sqrt(n)

        gamma = torch.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2) * sigma
        noise = torch.randn(size=y.shape(), device=device)
        
        y = y + h * d + gamma * noise

        if stride > 0 and t % stride == 0:
            print('iter %d, sigma %.2f', (t, sigma))
            all_ys.append(y.cpu().numpy())

        t += 1

    all_ys.append(y)
    return all_ys