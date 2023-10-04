import argparse
import torch
import numpy as np
import json
from utils.dataset import DataSet
from models.denoiser import Denoiser
from inverse.lnopt import run_optim, gnl_pca

# run argument parser
def args():
    parser = argparse.ArgumentParser(description='Denoiser Training')

    # option/mode for the script
    parser.add_argument('-f',
                        required=False,
                        type=str,
                        help='jupyter notebook')
    parser.add_argument('--mode',
                        required=False,
                        type=str,
                        help='script mode')
    parser.add_argument('--model_path',
                        type=str,
                        default='./assets/conv3_ln.pt')
    parser.add_argument('--pbar',
                        type=bool,
                        default=False)

    # arguments for optmization
    parser.add_argument('--batch_size',
                        type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=64,
                        help='number of epochs to train')
    parser.add_argument('--lr',
                        type=float,
                        default=5e-4)
    parser.add_argument('--decay_rate',
                        type=float,
                        default=0.90)
    parser.add_argument('--loss_type',
                        type=str,
                        default='MSE')
    parser.add_argument('--n_sample',
                        type=int,
                        default=64)
    parser.add_argument('--recon_method',
                        type=str,
                        default='denoiser')

    # see dataset.py for parameters for individual dataset
    parser.add_argument('--dataset',
                        type=str,
                        default='islvrc48')
    parser.add_argument('--data_path',
                        type=str,
                        default='islvrc')
    parser.add_argument('--linear',
                        type=bool,
                        default=True)
    parser.add_argument('--patch_size',
                        default=(48, 48))
    parser.add_argument('--test_size',
                        default=(48, 48))
    parser.add_argument('--scales',
                        default=[0.40, 0.20, 0.125])
    parser.add_argument('--test_scale',
                        default=[0.40, 0.20, 0.125])

    # network architecture
    parser.add_argument('--padding',
                        type=int,
                        default=1)
    parser.add_argument('--kernel_size',
                        type=int,
                        default=3)
    parser.add_argument('--num_kernels',
                        type=int,
                        default=64)
    parser.add_argument('--num_layers',
                        type=int,
                        default=20)
    parser.add_argument('--im_channels',
                        type=int,
                        default=3)

    # parse arguments and check
    args, _ = parser.parse_known_args()
    return args

args = args()
dict_args = vars(args)

# load dataset settings
keys = ['data_path', 'patch_size', 'test_size', 'scales',
        'test_scale', 'model_path', 'padding', 'kernel_size']

with open("data_config.json", "r") as fl:
    data = json.load(fl)
    arg_vals = data[args.dataset]

    # set up parameters for loading the dataset
    for idx in range(len(keys)):
        dict_args[keys[idx]] = arg_vals[idx]

# list all arguments and values
config_str = ' '.join(f'{k}={v}' for k, v in vars(args).items())

# load training and test set
data = DataSet.load_dataset(args)
train_set = torch.from_numpy(data.train_set())
test_set = data.test_set()

np.random.seed(0)
np.random.shuffle(test_set)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_torch = torch.tensor(test_set).permute([0, 3, 1, 2]).to(device)

# setup save name
save_name = args.data_path + str(args.patch_size[0])
paras = [args.n_sample, args.loss_type, args.batch_size,
            args.n_epoch, args.lr, args.decay_rate, args.pbar]

if args.recon_method == 'denoiser':
    # load denoiser model
    model = Denoiser(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    # run optimization
    run_optim(train_set, test_torch, model, save_name, config_str, *paras)

elif args.recon_method == 'linear':
    # run optimization
    gnl_pca(train_set, test_torch, save_name, config_str, *paras)