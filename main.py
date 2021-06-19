import argparse, torch, os, torch.multiprocessing as mp
from random import randint
from models.denoiser import Denoiser
from utils.training import train_denoiser, train_parallel
from utils.dataset import ISLVRC, test_model

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
                        default='./assets/linear_06-17-06.pt')

    # arguments for network training
    parser.add_argument('--batch_size', 
                        type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--n_epoch', 
                        type=int,
                        default=60,
                        help='number of epochs to train')
    parser.add_argument('--noise_level',
                        default=[1, 128])
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    parser.add_argument('--decay_epoch',
                        default=[40, 50])
    parser.add_argument('--decay_rate',
                        type=float,
                        default=0.50)
    parser.add_argument('--ddp',
                        type=bool,
                        default=False,
                        help='Distributed Data Parallel')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./assets/model_para.pt')
    
    # training dataset
    parser.add_argument('--patch_size', 
                        default= (48, 48))
    parser.add_argument('--test_size', 
                        default= (128, 128))
    parser.add_argument('--scales',
                        default=[1.0, 0.80, 0.60, 0.40, 0.20])
    parser.add_argument('--test_scale',
                        default=[0.5])
    parser.add_argument('--linear',
                        default=True)
    
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
    # train with DDP and Multi-GPUs
    if args.ddp:
        world_size = torch.cuda.device_count()
        print('start training with %d GPUs' % world_size)

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        args.seed = randint(0, 65535)
        mp.spawn(train_parallel, nprocs=world_size, 
        args=(world_size, args))

    # train with single GPU (or CPUs)
    else:
        # denoiser conv net
        model = Denoiser(args)
        print('number of parameters is ', 
            sum(p.numel() for p in model.parameters()))

        # load dataset
        print('load training data')
        islvrc = ISLVRC(args)
        
        print('start training')
        model = train_denoiser(islvrc.train_set(), 
                islvrc.test_set(), model, args)

        # save trained model
        print('save model parameters')
        torch.save(model.state_dict(), args.save_dir)

def test(args):
    test_set = ISLVRC(args, test_mode=True).test_set()

    # load denoiser
    model = Denoiser(args)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    input_psnr = []
    output_psnr = []
    for noise in range(110, 0, -10):
        psnr = test_model(test_set, model, noise, device)[0].mean(axis=1)
        input_psnr.append(psnr[0])
        output_psnr.append(psnr[1])

    print('input psnr: ', ['%.2f' % val for val in input_psnr])
    print('output psnr: ', ['%.2f' % val for val in output_psnr])

    return (input_psnr, output_psnr)    

if __name__ == '__main__':
    if args.mode == 'train':
        train(args)
    
    if args.mode == 'test':
        test(args)