"""
Bayesian Image Reconstruction Methods
    - Gaussian / Sparse MAP
    - Gaussian likelihood
"""
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
    def __init__(self, device, render, basis, mu, stride=4):
        self.render = torch.tensor(render, device=device)
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

    def _conv_basis(this, image):
        '''
        Compute the prior value with convolutional
        projection onto a set of basis kernels
            - image: (batch_size, 3, n, n)
        '''
        return conv2d(image, this.kernel, this.bias, this.stride, 'valid')

    @abstractmethod
    def conv_prior(this, image):
        '''
        Loss value associated with the prior (Gaussian and Sparse)
            - image: (batch_size, 3, n, n)
        '''
        pass

    def llhd(this, msmt, image):
        '''
        Compute the likelihood of the estimate (image)
            - msmt: (batch_size, n_measurements)
            - image: (batch_size, image_size)
        '''
        diff = (this.render @ image.T).T - msmt
        return 0.5 * torch.pow(diff, 2).sum(1)

class GaussianEstimator(BayesEstimator):
    def __init__(self, device, basis, mu, stride=4):
        super().__init__(device, basis, mu, stride)

    # Gauassian prior
    def conv_prior(this, image):
        coeff = this._conv_basis(image)
        coeff = torch.pow(coeff, 2).reshape(coeff.shape[0], -1)

        return 0.5 * coeff.sum(1)

class SparseEstimator(BayesEstimator):
    def __init__(self, device, basis, mu, stride=4):
        super().__init__(device, basis, mu, stride)

    # Sparse prior
    def conv_prior(this, image):
        coeff = this._conv_basis(image)
        coeff = coeff.reshape(coeff.shape[0], -1)

        return torch.abs(coeff).sum(1)