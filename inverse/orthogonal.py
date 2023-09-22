from inverse.solver import RenderMatrix
from torch.linalg import vector_norm as vnorm
import torch.nn.utils.parametrizations as para
import torch, numpy as np
import torch.nn as nn

class OrthMatrix(RenderMatrix):
    """
    Define a measurement matrix that is orthgonal
    using householder parameterization
    """
    def __init__(self, n_sample, im_size, device):
        n_pixel = np.prod(im_size)

        # init orthgonal matrix with householder product parameterization
        linear = torch.nn.Linear(n_pixel, n_sample).to(device)
        self.para = para.orthogonal(linear, orthogonal_map='householder')

        super().__init__(self.para.weight, im_size, device)

    def forward(self):
        # generate the measurement matrix based on the parameterization
        self.R = self.para.weight

class LinearInverse(nn.Module):
    """
    Implement the linear inverse reconstruction produce as a nn.Module object,
    with the sample - recon calculation as the forward function.

    The measurement matrix is part of the model, and is assumed to be orthogonal.

    This enables us to perform reconstruction on mini-batch and multi-GPU training.
    """

    def __init__(self, n_sample, im_size, denoiser):
        super().__init__()

        # save variables
        self.model = denoiser
        self.im_size = im_size
        self.n_pixel = np.prod(im_size)
        self.n_sample = n_sample
        self.h_increase = True

        # no grad flag for the denoiser model
        for param in self.model.parameters():
            param.requires_grad = False

        # initialize an orthogonal linear measurement matrix
        linear = torch.nn.Linear(self.n_pixel, self.n_sample)
        self.linear = para.orthogonal(linear, orthogonal_map='householder')
        self.mtx = self.linear.weight

        # default parameter for the reconstruction
        self.h_init = 0.10
        self.beta = 0.10
        self.sig_end = 0.01
        self.max_t = 100
        self.last_t = None

    def refresh(self):
        self.mtx = self.linear.weight

    def assign(self, mtx):
        """
        Assign a measurement matrix. If the matrix is not orthogonal,
        its orthogonal component will be obtained using QR decomposition.
        """
        self.linear.weight = mtx
        self.refresh()

        return self

    def to(self, device):
        return_val = super().to(device)
        self.refresh()
        return return_val

    def log_grad(self, x):
        return - self.model(x)

    def measure(self, x):
        """
        x: images of size [N, C, W, H]

        Note a transpose is required due to different
        in convention between MATLAB and torch
        (Row-Major vs Column-Major), and here we are
        using the MATLAB convention for compatibility
        """
        return torch.matmul(x.transpose(2, 3).flatten(start_dim=1), self.mtx.t())

    def recon(self, m):
        """
        m: vectors of size [N, M]
        """
        new_shape = [-1, self.im_size[0], self.im_size[2], self.im_size[1]]
        return torch.matmul(m, self.mtx).reshape(new_shape).transpose(2, 3)

    def inverse(self, msmt):
        """
        msmt: measurements of images

        Perform denoiser reconstruction on
        images based on linear measurements
        """
        # measurement matrix calculation
        M = self.measure
        M_T = self.recon

        # init variables
        proj = M_T(msmt)
        e = torch.ones_like(proj)
        n = torch.numel(e[0])
        mu = 0.5 * (e - M_T(M(e))) + proj
        y = torch.randn_like(mu) + mu
        sigma = vnorm(self.log_grad(y),
                dim=(1, 2, 3)) / np.sqrt(n)

        # crose-to-fine reconstruction
        t = 1
        # stop criteria (mean) for batch input
        while torch.mean(sigma) > self.sig_end:
            # update step size
            h = self.h_init
            if self.h_increase:
                h = (self.h_init * t) / (1 + self.h_init * (t - 1))

            # projected log prior gradient
            d = self.log_grad(y)
            d = (d - M_T(M(d)) + proj - M_T(M(y)))

            # noise magnitude
            sigma = vnorm(d, dim=(1, 2, 3)) / np.sqrt(n)

            # injected noise
            gamma = np.sqrt((1 - self.beta * h) ** 2 - (1 - h) ** 2) * sigma
            noise = torch.randn_like(y)
            # expand gamma for shape matching
            gamma = gamma[:, None, None, None].repeat([1, *self.im_size])

            # update image
            y = y + h * d + gamma * noise
            t += 1

            # safe guard for iteration limit (GPU memory limit)
            # (typically) use in conjection with grad=True
            if t > self.max_t:
                break

        self.last_t = t

        # run a final denoise step and return the results
        return y + self.log_grad(y)

    def forward(self, x):
        """
        x: images of size [N, C, W, H]

        Perform denoiser reconstruction on
        images based on linear measurements
        """
        # update the measurement matrix
        # based on the parameterization
        self.refresh()

        # compute the linear measurement
        msmt = self.measure(x)

        # run the reconstruction routine
        return self.inverse(msmt)

class LinearProjection(nn.Module):
    '''
    Linear projection with orthogonalized measurement matrix
    '''

    def __init__(self, n_sample, im_size):
        super().__init__()

        self.im_size = im_size
        self.n_pixel = np.prod(im_size)
        self.n_sample = n_sample

        # initialize an orthogonal linear measurement matrix
        linear = torch.nn.Linear(self.n_pixel, self.n_sample)
        self.linear = para.orthogonal(linear, orthogonal_map='householder')

    def forward(self, x):
        # measurement matrix
        proj_mtx = self.linear.weight

        # compute projection
        x_flat = x.transpose(2, 3).flatten(1)
        recon_flat = x_flat @ proj_mtx.t() @ proj_mtx

        new_shape = [-1, self.im_size[0], self.im_size[2], self.im_size[1]]
        return recon_flat.reshape(new_shape).transpose(2, 3)