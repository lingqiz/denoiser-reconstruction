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
        torch.nn.init.uniform_(linear.weight, a=0.0, b=1.0)
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

        # no grad flag for the denoiser model
        for param in self.model.parameters():
            param.requires_grad = False

        # initialize an orthogonal linear measurement matrix
        linear = torch.nn.Linear(self.n_pixel, self.n_sample)
        torch.nn.init.uniform_(linear.weight, a=0.0, b=1.0)

        # save the orthogonal linear layer
        self.linear = para.orthogonal(linear, orthogonal_map='householder')
        self.mtx = self.linear.weight

        # default parameter for the reconstruction
        self.h_init = 0.10
        self.beta = 0.10
        self.sig_end = 0.01
        self.t_max = 100

    def refresh(self):
        self.mtx = self.linear.weight

    def assign(self, mtx):
        """
        Assign a measurement matrix. If the matrix is not orthogonal,
        its orthogonal component will be obtained using QR decomposition.
        """
        self.linear.weight = mtx
        self.refresh()

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
        new_shape = [-1, *self.im_size]
        return torch.matmul(m, self.mtx).reshape(new_shape).transpose(2, 3)

    def forward(self, x):
        """
        x: images of size [N, C, W, H]
        """
        # update the measurement matrix
        # based on the parameterization
        self.refresh()

        # measurement matrix calculation
        M = self.measure
        M_T = self.recon

        # init variables
        proj = M_T(M(x))
        e = torch.ones_like(proj)
        n = torch.numel(e[0])
        mu = 0.5 * (e - M_T(M(e))) + proj
        y = torch.randn_like(mu) + mu
        sigma = vnorm(self.log_grad(y),
                dim=(1, 2, 3)) / np.sqrt(n)

        # start the iterative procedure
        t = 1
        while torch.min(sigma) > self.sig_end:
            # update step size
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
            if t > self.t_max:
                break

        # run a final denoise step and return the results
        final = y + self.log_grad(y)
        return final, torch.tensor(t)

