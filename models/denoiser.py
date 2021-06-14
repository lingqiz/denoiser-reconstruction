import torch
import torch.nn as nn
import numpy as np

class Denoiser(nn.Module):
    """
    A simple CNN for image denoising

    The additive bias term is removed from the network, 
    including both the convolutional and batch normalization layers.
    """ 

    def __add_layer(self):
        self.conv_layers.append(
            nn.Conv2d(self.num_channels,self.num_kernels, self.kernel_size, padding=self.padding, bias=False))

    def __init__(self, args):
        super().__init__()
        
        # parameters for the convolution 
        self.padding = args.padding
        self.num_kernels = args.num_kernels
        self.kernel_size = args.kernel_size
        self.num_layers = args.num_layers
        self.num_channels = args.num_channels

        self.relu = nn.ReLU(inplace=True)
        self.conv_layers = nn.ModuleList([])
        self.running_sd = nn.ParameterList([])
        self.gammas = nn.ParameterList([])

        # add conv layers
        self.__add_layer()        

        for idx in range(self.num_layers - 2):
            self.__add_layer()
            
            # approximate Batch Normalization without the additive bias term
            self.running_sd.append(
                nn.Parameter(torch.ones(1, self.num_kernels, 1, 1), requires_grad=False))
            g = (torch.randn((1, self.num_kernels, 1, 1)) * (2. / 9. / 64.)).clamp_(-0.025, 0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True))
            
        # last layer without BN
        self.__add_layer()

    def forward(self, x):
        for idx, conv in zip(range(self.num_layers, self.conv_layers)):
            if idx == self.num_layers - 1:
                return conv(x)

            x = conv(x)            
            if idx > 0:
                # apply BN
                sd_x = torch.sqrt(x.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-05)
                if self.training:                    
                    x = x / sd_x.expand_as(x)
                    self.running_sd[idx - 1].data = (1 - .1) * self.running_sd[idx - 1].data + .1 * sd_x
                    x = x * self.gammas[idx - 1].expand_as(x)
                else:
                    x = x / self.running_sd[idx - 1].expand_as(x)
                    x = x * self.gammas[idx - 1].expand_as(x)

            x = self.relu(x)
                