import torch, numpy as np

class RenderMatrix:
    def __init__(self, R, im_size, device):
        self.im_size = im_size
        self.R = R.to(device)

    def measure(self, x):
        '''
        Given (orthogonalized) render matrix R 
        and image x, compute the measurement
        '''    
        return torch.matmul(self.R, x.permute([1, 2, 0]).flatten())

    def recon(self, msmt):
        '''
        From measurement to image space
        (projection onto R)
        '''
        return torch.matmul(self.R.T, msmt).reshape(self.im_size).permute([2, 0, 1])

# sample from prior with linear constraint (render matrix)
def linear_inverse(model, render, msmt, h_init=0.01, beta=0.01, sig_end=0.01, stride=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
    
    # helper function for pytorch image to numpy image
    numpy_image = lambda y: y.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # the network calculates the noise residual
    def log_grad(y):
        with torch.no_grad():
            return - model(y)

    R = render.measure
    R_T = render.recon

    # init variables
    e = torch.ones_like(R_T(msmt))
    n = torch.numel(e)

    mu = 0.5 * (e - R_T(R(e))) + R_T(msmt)         
    y = torch.normal(mean=mu, std=1.0).unsqueeze(0).to(device)

    sigma = torch.norm(log_grad(y)) / np.sqrt(n)

    t = 1
    all_ys = []
    while sigma > sig_end:
        # update step size
        h = (h_init * t) / (1 + h_init * (t - 1))

        # projected log prior gradient
        d = log_grad(y).squeeze(0)
        d - R_T(R(d))
        d = (d - R_T(R(d)) + R_T(msmt) - 
            R_T(R(y.squeeze(0)))).unsqueeze(0)

        # inject noise
        sigma = torch.norm(d) / np.sqrt(n)
        gamma = np.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2) * sigma
        noise = torch.randn(size=y.size(), device=device)

        # update image        
        y = y + h * d + gamma * noise

        if stride > 0 and (t - 1) % stride == 0:
            print('iter %d, sigma %.2f' % (t, sigma.item()))
            all_ys.append(numpy_image(y))

        t += 1
    
    all_ys.append(numpy_image(y + log_grad(y)))
    return all_ys

        

