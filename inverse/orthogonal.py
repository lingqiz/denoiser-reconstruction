# reconstruction and optimization with orthogonal measurement matrix
from inverse.solver import RenderMatrix, linear_inverse
import torch.nn.utils.parametrizations as para
import torch, numpy as np

class OrthMatrix(RenderMatrix):
    def __init__(self, n_sample, im_size, device):
        n_pixel = np.prod(im_size)
        
        # init orthgonal matrix with householder product parameterization
        linear = torch.nn.Linear(n_pixel, n_sample).to(device)
        torch.nn.init.uniform_(linear.weight, a=0.0, b=1.0)
        self.para = para.orthogonal(linear, orthogonal_map='householder')
    
        super().__init__(self.para.weight, im_size, device)
        
    def _update(self):
        # update the measurement matrix based on the parameterization
        self.R = self.para.weight

