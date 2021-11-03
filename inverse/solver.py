import torch, numpy as np
from abc import ABC, abstractmethod
import warnings

# base class for measurement matrix
class Measurement(ABC):
    @abstractmethod
    def measure(self, x):
        pass

    @abstractmethod
    def recon(self, x):
        pass

# simple measurement matrix with convolution
# use ConvTranspose2d and Conv2d for projection
class ConvMatrix(Measurement):
    def __init__(self, kernel_size, stride, device, channels=3):
        # use to record image size
        self.imsize = None

        # conv_transpose work for stride >= kernel_size
        if stride < kernel_size:
            raise Warning(f'''Stride ({stride}) should NOT be smaller than kernel size ({kernel_size})
                                for ConvTranspose2d to act properly as a linear projection''')

        # sampling kernel
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size, stride,
                                    groups=channels, bias=False, device=device)

        # inverse operation
        self.conv_tr = torch.nn.ConvTranspose2d(channels, channels, kernel_size, stride,
                                                groups=channels, bias=False, device=device)

        # make an averaging conv kernal
        kernel = torch.ones(channels, 1, kernel_size, kernel_size, device=device)
        kernel = torch.nn.Parameter(kernel / torch.norm(kernel[0, 0, :, :]))

        self.conv.weight = kernel
        self.conv_tr.weight = kernel

    # assume calculation requires no gradient
    # linear measurement
    def measure(self, x):
        with torch.no_grad():
            self.imsize = x.shape[1:]
            x = self.conv(x.unsqueeze(0))
            return x.squeeze(0)

    # linear projection
    def recon(self, x):
        with torch.no_grad():
            x = x.unsqueeze(0)
            x = self.conv_tr(x, output_size=self.imsize)
            return x.squeeze(0)

class RenderMatrix(Measurement):
    def __init__(self, R, im_size, device):
        self.im_size = im_size
        self.R = R.to(device)

    def measure(self, x):
        '''
        Given (orthogonalized) render matrix R
        and image x, compute the measurement

        A transpose is required due to different
        in convention between MATLAB and torch
        '''

        return torch.matmul(self.R, x.transpose(1, 2).flatten())

    def recon(self, msmt):
        '''
        From measurement to image space
        '''

        return torch.matmul(self.R.T, msmt).reshape(self.im_size).transpose(1, 2)

class ArrayMatrix(Measurement):
    '''
    Generalization of the RenderMatrix class to an array
    of matrices that tile through larger images
    '''

    def __init__(self, array, array_size, im_size, device):
        self.device = device
        self.nx = array_size[0]
        self.ny = array_size[0]

        # im_size[1] == im_size[2]
        self.im_size = im_size
        self.edge = im_size[1]

        for idx in range(self.nx):
            for idy in range(self.ny):
                array[idx][idy] = torch.tensor(array[idx][idy].astype('float32')).to(device)

        self.array = array

    def measure(self, x):
        '''
        Measurement array from a set of matrices
        '''

        # init
        msmt = [[0 for y in range(self.ny)]
                      for x in range(self.nx)]

        # loop through matrices
        for idx in range(self.nx):
            for idy in range(self.ny):
                sliced = x[:, idy * self.edge : (idy + 1) * self.edge,
                          idx * self.edge : (idx + 1) * self.edge]

                msmt[idx][idy] = \
                    torch.matmul(self.array[idx][idy],
                                 sliced.transpose(1, 2).flatten())
        return msmt

    def recon(self, msmt):
        '''
        From an array of measurements to image space
        '''

        # init
        recon = torch.empty(size=(3, self.ny * self.edge,
                                  self.nx * self.edge),
                            device=self.device)

        # loop through measurements
        for idx in range(self.nx):
            for idy in range(self.ny):
                sliced = torch.matmul(self.array[idx][idy].T, msmt[idx][idy])

                recon[:, idy * self.edge : (idy + 1) * self.edge,
                      idx * self.edge : (idx + 1) * self.edge] = \
                sliced.reshape(self.im_size).transpose(1, 2)

        return recon

# sample from prior with linear constraint (render matrix)
def linear_inverse(model, render, input, h_init=0.01, beta=0.01, sig_end=0.01,
                    t_max=float('inf'), stride=10, seed=None, with_grad=False):
    if not (seed is None):
        torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)

    # helper function for pytorch image to numpy image
    numpy_image = lambda y: y.detach().cpu() \
                .squeeze(0).permute(1, 2, 0).numpy()

    # the network calculates the noise residual
    if with_grad:
        log_grad = lambda y: - model(y)
    else:
        def log_grad(y):
            with torch.no_grad():
                return - model(y)

    # measurement matrix calculation
    R = render.measure
    R_T = render.recon

    # init variables
    if input.dim() == 1:
        proj = R_T(input)
    elif input.dim() == 3:
        proj = R_T(R(input))

    e = torch.ones_like(proj)
    n = torch.numel(e)

    mu = 0.5 * (e - R_T(R(e))) + proj
    y = torch.normal(mean=mu, std=1.0).unsqueeze(0).to(device)
    sigma = torch.norm(log_grad(y)) / np.sqrt(n)

    t = 1
    all_ys = []
    while sigma > sig_end:
        # update step size
        h = (h_init * t) / (1 + h_init * (t - 1))

        # projected log prior gradient
        d = log_grad(y).squeeze(0)
        d = (d - R_T(R(d)) + proj -
            R_T(R(y.squeeze(0)))).unsqueeze(0)

        # noise magnitude
        sigma = torch.norm(d) / np.sqrt(n)

        # protect against divergence
        div_thld  = 1e2
        iter_thld = 1e3
        if sigma > div_thld or t > iter_thld:
            warnings.warn('Divergence detected, resample with \
                larger step size and tolerance.', RuntimeWarning)

            return linear_inverse(model, render, input,
                                 h_init, beta * 2, sig_end * 2,
                                 t_max, stride, seed, with_grad)

        # inject noise
        gamma = np.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2) * sigma
        noise = torch.randn(size=y.size(), device=device)

        # update image
        y = y + h * d + gamma * noise

        if stride > 0 and (t - 1) % stride == 0:
            print('iter %d, sigma %.2f' % (t, sigma.item()))
            all_ys.append(numpy_image(y))

        t += 1

        # safe guard for iteration limit
        # (typically) use in conjection with grad=True
        # for GPU memory limit
        if t > t_max:
            break

    final = y + log_grad(y)
    all_ys.append(numpy_image(final))

    if with_grad:
        return final.squeeze(0), t, all_ys

    return all_ys

