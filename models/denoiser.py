import torch
import torch.nn as nn
import numpy as np

class Denoiser(nn.Module):
    """
    A simple CNN for image denoising.
    The network takes noisy images as input and returns residual.

    The additive bias term is removed from the network, 
    including both the convolutional and batch normalization layers.

    See https://arxiv.org/abs/1906.05478
    """ 

    def __init__(self, args):
        super().__init__()
        
        # parameters for the convolution 
        self.padding = args.padding
        self.num_kernels = args.num_kernels
        self.kernel_size = args.kernel_size
        self.num_layers = args.num_layers
        self.im_channels = args.im_channels

        # network layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_layers = nn.ModuleList([])
        
        # batch norm
        self.gammas = nn.ParameterList([])
        self.register_buffer('running_sd', 
            torch.empty([self.num_layers - 2, 1, self.num_kernels, 1, 1]))

        # add conv layers
        self.__add_layer(ch_in=self.im_channels, ch_out=self.num_kernels)

        for idx in range(self.num_layers - 2):
            self.__add_layer(ch_in=self.num_kernels, ch_out=self.num_kernels)
            
            # approximate Batch Normalization without the additive bias term
            self.running_sd[idx, ] = torch.ones(1, self.num_kernels, 1, 1)

            g = (torch.randn((1, self.num_kernels, 1, 1)) * (2. / 9. / 64.)).clamp_(-0.025, 0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True))
                        
        # last layer without BN
        self.__add_layer(ch_in=self.num_kernels, ch_out=self.im_channels)
        
        # weight init
        self.__weight_init()

    # helper functions
    def __add_layer(self, ch_in, ch_out):
        self.conv_layers.append(
            nn.Conv2d(ch_in, ch_out, self.kernel_size, padding=self.padding, bias=False))

    def __weight_init(self):

        def init_fun(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')

        self.apply(init_fun)

    # forward function
    def forward(self, x):
        # loop through all the layers
        for idx, conv in zip(range(self.num_layers), self.conv_layers):
            if idx == self.num_layers - 1:
                return conv(x)

            x = conv(x)
            if idx > 0:
                # apply BN
                sd_x = torch.sqrt(x.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-05)
                if self.training:                    
                    x = x / sd_x.expand_as(x)
                    self.running_sd[idx - 1] = (1 - 0.1) * self.running_sd[idx - 1] + 0.1 * sd_x
                    x = x * self.gammas[idx - 1].expand_as(x)
                else:
                    x = x / self.running_sd[idx - 1].expand_as(x)
                    x = x * self.gammas[idx - 1].expand_as(x)

            x = self.relu(x)