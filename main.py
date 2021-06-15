import numpy as np
import torch
import torch.nn as nn 
import argparse

from models.denoiser import Denoiser

# run argument parser
def args():
    parser = argparse.ArgumentParser(description='Denoiser Training')
    parser.add_argument('-f',
                        required=False,
                        type=str,
                        help='jupyter notebook')
    parser.add_argument('--mode',
                        required=False,
                        type=str,
                        help='script mode')

    # arguments for network training
    parser.add_argument('--batch_size', 
                        type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--n_epoch', 
                        type=int, 
                        default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--noise_level',
                        default=100)
    
    # training dataset
    parser.add_argument('--patch_size', 
                        default= (64, 64))
    parser.add_argument('--scales',
                        default=[0.90, 0.75, 0.50, 0.25])    
    
    # network architecture
    parser.add_argument('--padding', 
                        default= 1)
    parser.add_argument('--kernel_size', 
                        default= 3)                        
    parser.add_argument('--num_kernels', 
                        default= 64)
    parser.add_argument('--num_layers', 
                        default= 20)
    parser.add_argument('--im_channels', 
                        default= 3)

    # parse arguments and check
    args = parser.parse_args()
    return args

args = args()