import argparse, torch
from models.denoiser import Denoiser
from utils.training import train_denoiser
from utils.dataset import ISLVRC

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
                        type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--n_epoch', 
                        type=int, 
                        default=20,
                        help='number of epochs to train')
    parser.add_argument('--noise_level',
                        default=[1, 100])
    parser.add_argument('--lr',
                        default=2e-3)
    parser.add_argument('--lr_decay',
                        default=0.8)
    parser.add_argument('--save_dir',
                        './assets/model_para.pt')
    
    # training dataset
    parser.add_argument('--patch_size', 
                        default= (64, 64))
    parser.add_argument('--scales',
                        default=[1.0, 0.75, 0.50, 0.25])
    
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

def train(args):
    # load dataset
    islvrc = ISLVRC(args)

    # denoiser CNN
    model = Denoiser(args)
    print('number of parameters is ', 
        sum(p.numel() for p in model.parameters()))

    # model training 
    model = train_denoiser(islvrc.train_set(), 
            islvrc.test_set(), model, args)

    # save trained model
    torch.save(model.state_dict(), args.save_dir)

if __name__ == '__main__':
    if args.mode == 'train':
        train(args)

    if args.mode == 'test':
        pass