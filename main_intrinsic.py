from random import randint
from models.denoiser import Denoiser
from utils.training import train_denoiser, train_parallel
from utils.dataset import DataSet, test_model
import argparse, torch, os, torch.multiprocessing as mp

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
                        default='./assets/conv3_intr.pt')

    # arguments for network training
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=100,
                        help='number of epochs to train')
    parser.add_argument('--noise_level',
                        default=[0, 512])
    parser.add_argument('--lr',
                        type=float,
                        default=5e-4)
    parser.add_argument('--decay_epoch',
                        default=[40, 60, 80, 100])
    parser.add_argument('--decay_rate',
                        type=float,
                        default=0.50)
    parser.add_argument('--ddp',
                        type=bool,
                        default=True,
                        help='Distributed Data Parallel')
    parser.add_argument('--save_path',
                        type=str,
                        default='./assets/conv3_intrinsic.pt')

    # see dataset.py for parameters for individual dataset
    parser.add_argument('--data_path',
                        type=str,
                        default='intrinsic')
    parser.add_argument('--data_range',
                        type=int,
                        default=10)
    parser.add_argument('--linear',
                        type=bool,
                        default=True)
    parser.add_argument('--patch_size',
                        default=None)
    parser.add_argument('--test_size',
                        default=None)
    parser.add_argument('--scales',
                        default=None)
    parser.add_argument('--test_scale',
                        default=None)

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
                        default=6)

    # parse arguments and check
    args, _ = parser.parse_known_args()
    return args

args = args()

def train(args):
    # load dataset
    print('start loading training data')

    dataset = DataSet.load_dataset(args)
    train_set = torch.from_numpy(dataset.train_set())
    test_set  = torch.from_numpy(dataset.test_set())

    print('dataset loaded, size %d' % train_set.size()[0])

    # train with DDP and Multi-GPUs
    if args.ddp:
        # move dataset to shared memory
        train_set.share_memory_()
        test_set.share_memory_()

        world_size = torch.cuda.device_count()
        print('model training with %d GPUs' % world_size)

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        args.seed = randint(0, 65535)
        mp.spawn(train_parallel, nprocs=world_size,
        args=(world_size, train_set, test_set, args))

    # train with single GPU (or CPUs)
    else:
        # denoiser conv net
        model = Denoiser(args)
        print('number of parameters is ',
            sum(p.numel() for p in model.parameters()))

        print('start training')
        model = train_denoiser(train_set, test_set, model, args)

        # save trained model
        print('save model parameters')
        torch.save(model.state_dict(), args.save_path)

def test(args):
    test_set = DataSet.load_dataset(args, test_mode=True).test_set()

    # load denoiser
    model = Denoiser(args)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    input_psnr = []
    output_psnr = []

    if args.data_path == 'intrinsic':
        noise_level = 512.0
        data_range = args.data_range
        clip_range = (-5.5, 5.5)
    else:
        noise_level = 128.0
        data_range = None
        clip_range = (0, 1)

    for noise in range(noise_level, 0, -10):
        psnr = test_model(test_set, model, noise=noise, device=device,
            data_range=data_range, clip_range=clip_range)[0].mean(axis=1)
        
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
