"""
Bayesian Image Reconstruction Methods
    - Gaussian / Sparse MAP
    - Gaussian likelihood
"""
from pickletools import optimize
import torch, numpy as np
from torch.nn.functional import conv2d
from abc import ABC, abstractmethod

class BayesEstimator(ABC):
    """
    Base class for Bayesian reconstruction methods

    variables render, basis, and mu are numpy types
        - render: (n_measurements, image_size)
        - basis: (out_channels, image_size)
        - mu: (image_size, 1)
    """
    def __init__(self, device, render, basis, mu, lbda=1e-4, stride=4):
        self.render = torch.tensor(render, device=device)
        self.device = device
        self.lbda = lbda
        self.stride = stride

        self.bias = torch.tensor(-(basis @ mu),
        dtype=torch.float32).to(device).squeeze()

        # basis function as image kernel
        n_dim = int(np.sqrt(basis.shape[1] / 3))
        k_size = (n_dim, n_dim, 3)

        all_basis = np.zeros([basis.shape[0], *k_size])
        for idx in range(basis.shape[0]):
            # basis are built using matlab fortran convention
            kernel = basis[idx, :].reshape(k_size, order='F')
            all_basis[idx] = kernel

        # build the weight kernel
        self.kernel = torch.tensor(all_basis.transpose([0, 3, 1, 2]),
                                   dtype=torch.float32).to(device)

    @staticmethod
    def measure(render, image):
        '''
        matrix - image vector product
            - render: (n_measurements, image_size)
            - image: (batch_size, 3, n, n)
        '''
        # transpose is needed for flatten to result in
        # column-major convention (MATLAB / Fortran)
        return image.transpose(2, 3).flatten(1) @ render.t()

    def _conv_basis(self, image):
        '''
        Compute the prior value with convolutional
        projection onto a set of basis kernels
            - image: (batch_size, 3, n, n)
        '''
        return conv2d(image, self.kernel, self.bias, self.stride, 'valid')

    @abstractmethod
    def conv_prior(self, image):
        '''
        Loss value associated with the prior (Gaussian and Sparse)
            - image: (batch_size, 3, n, n)
        '''
        pass

    def neg_llhd(self, msmt, image):
        '''
        Compute the likelihood of the estimate (image)
            - msmt: (batch_size, n_measurements)
            - image: (batch_size, 3, n, n)
        '''
        diff = self.measure(self.render, image) - msmt
        return 0.5 * torch.pow(diff, 2).sum(1)

    def objective(self, msmt, image):
        '''
        Compute the objective function
            - msmt: (batch_size, n_measurements)
            - image: (batch_size, 3, n, n)
        '''
        return self.neg_llhd(msmt, image) + self.lbda * self.conv_prior(image)

    def recon(self, msmt, im_size, n_iter=100, lr=1e-2):
        '''
        Reconstruct image(s) from measurements
            - msmt: (batch_size, n_measurements)
            - im_size: (batch_size, 3, n, n)
        '''
        init = torch.rand(im_size, type=torch.float32,
                device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([init], lr=lr)
        for iter in range(n_iter):
            optimizer.zero_grad()
            obj = self.objective(msmt, init).sum()

            obj.backward()
            optimizer.step()

            if iter % 5 == 0:
                print('Iteration: {}, Objective: {}'.format(iter, obj.sum()))

        return init.detach()

class GaussianEstimator(BayesEstimator):
    def __init__(self, device, basis, mu, stride=4):
        super().__init__(device, basis, mu, stride)

    # Gauassian prior
    def conv_prior(self, image):
        coeff = self._conv_basis(image)
        coeff = torch.pow(coeff, 2).reshape(coeff.shape[0], -1)

        return 0.5 * coeff.sum(1)

class SparseEstimator(BayesEstimator):
    def __init__(self, device, basis, mu, stride=4):
        super().__init__(device, basis, mu, stride)

    # Sparse prior
    def conv_prior(self, image):
        coeff = self._conv_basis(image)
        coeff = coeff.reshape(coeff.shape[0], -1)

        return torch.abs(coeff).sum(1)