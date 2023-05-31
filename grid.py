# grid search on optimization hyper-parameters

from random import randint
from utils.training import train_parallel
from utils.dataset import DataSet
from utils.helper import parse_args
import torch, os, torch.multiprocessing as mp

# argument parsing
args = parse_args()

# grid search over parameters
# train with DDP and Multi-GPUs
def run_search(args):
    # print arg values
    print('training configuration: \n', args)

    # load dataset
    print('start loading training data')

    dataset = DataSet.load_dataset(args)
    train_set = torch.from_numpy(dataset.train_set())
    test_set  = torch.from_numpy(dataset.test_set())

    print('dataset loaded, size %d' % train_set.size()[0])

    # move dataset to shared memory
    train_set.share_memory_()
    test_set.share_memory_()

    world_size = torch.cuda.device_count()
    print('model training with %d GPUs' % world_size)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # list of learning rate and decay rate
    all_lrs = [1e-3, 1e-2, 1e-1]
    all_decay = [1e-2, 0.1, 1.0]

    # iterate through
    for idx in range(len(all_lrs)):
        for idy in range(len(all_decay)):
            # set parameter values
            args.lr = all_lrs[idx]
            args.decay_adam = all_decay[idy]

            # run network training
            args.seed = randint(0, 65535)
            mp.spawn(train_parallel, nprocs=world_size,
                        args=(world_size, train_set, test_set, args))

if __name__ == '__main__':
    # run grid search
    run_search(args)
