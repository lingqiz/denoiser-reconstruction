import torch, torch.nn as nn, time, datetime, random
from models.denoiser import Denoiser
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils.dataset import ISLVRC, test_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DSP

def sample_noise(size, noise_level):    
    noise = torch.empty(size=size)
    for idx in range(int(size[0])):
        noise_sd = random.randint(noise_level[0], noise_level[1]) / 255.0
        noise[idx] = torch.normal(mean=0.0, std=noise_sd, size=list(size[1:]))

    return noise

def train_run(model, train_set, test_set, sampler, rank, args):
    # setup for training
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, 
                milestones=args.decay_epoch, gamma=args.decay_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    # run training
    torch.backends.cudnn.benchmark = True
    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        if sampler:
            train_set.sampler.set_epoch(epoch)

        for count, batch in enumerate(train_set):
            optimizer.zero_grad(set_to_none=True)
            
            # images in torch are in [c, h, w] format
            batch = batch.permute(0, 3, 1, 2).contiguous().to(rank)
            noise = sample_noise(batch.size(), args.noise_level).to(rank)
            noise_input = batch + noise

            # auto mixed precision forward pass
            with autocast():
                # the network takes noisy images as input 
                # and returns residual (i.e., skip connections)
                residual = model(noise_input)
                loss = criterion(residual, noise)

            total_loss += loss.item()

            # backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        
        if rank == 0 or rank == 'cpu':
            # print some diagnostic information
            print('epoch %d/%d' % (epoch + 1, args.n_epoch))
            
            psnr = test_model(test_set, model, noise=128.0, device=rank)[0].mean(axis=1)
            print('average loss %.6f' % (total_loss / float(count)))
            print('test psnr in %.2f, out %.2f' % (psnr[0], psnr[1]))

            print('time elapsed: %s' % str(datetime.timedelta(
                seconds=time.time() - start_time))[:-4])

def train_denoiser(train_set, test_set, model, args):
    # training with GPU if available
    rank = (0 if torch.cuda.is_available() else 'cpu')
    model = model.to(rank)

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size, 
                shuffle=True, pin_memory=True)

    train_run(model, train_set, test_set, sampler=False, rank=rank, args=args)
    return model.eval().cpu()

# training with Distributed Data Parallel (process level parallelism)
def train_parallel(rank, world_size, args):    
    dist.init_process_group("nccl", init_method='env://', 
                            rank=rank, world_size=world_size)
    
    # wrap the model with DDP
    # In DDP, the constructor, the forward pass, 
    # and the backward pass are distributed synchronization points
    model = Denoiser(args).to(rank)    
    model = DDP(model, device_ids=[rank])

    # setup dataset
    islvrc = ISLVRC(args)
    train_set = islvrc.train_set()
    test_set = islvrc.test_set()

    # training dataset
    data_sampler = DSP(train_set, world_size, rank, shuffle=True, seed=args.seed)
    train_set = DataLoader(train_set, batch_size=args.batch_size, drop_last=True,
                shuffle=False, pin_memory=True, sampler=data_sampler)

    train_run(model, train_set, test_set, sampler=True, rank=rank, args=args)

    # save the parameters of the model
    if rank == 0:
        print('save model parameters')
        torch.save(model.module.state_dict(), args.save_dir)

    dist.destroy_process_group()