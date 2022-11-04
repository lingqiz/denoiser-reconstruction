from inverse.solver import RenderMatrix
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
        # save variables
        self.model = denoiser
        self.n_pixel = np.prod(im_size)
        self.n_sample = n_sample

        # initialize an orthogonal linear measurement matrix
        linear = torch.nn.Linear(self.n_pixel, self.n_sample)
        torch.nn.init.uniform_(linear.weight, a=0.0, b=1.0)

        self.linear = para.orthogonal(linear, orthogonal_map='householder')
        self.mtx = self.linear.weight

    def assign_mtx(self, mtx):
        """
        Assign a measurement matrix. If the matrix is not orthogonal,
        its orthogonal component will be obtained using QR decomposition.
        """
        self.linear.weight = mtx

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
