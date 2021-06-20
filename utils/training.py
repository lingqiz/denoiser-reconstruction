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

def train_denoiser(train_set, test_set, model, args):
    # training with GPU(s) if available
    # use DataParallel for multi-GPU training 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # setup for training
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, 
                milestones=args.decay_epoch, gamma=args.decay_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size, 
                shuffle=True, pin_memory=True)

    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for _, batch in enumerate(train_set):
            optimizer.zero_grad()

            # choose a noise level for the batch
            noise_level = random.randint(args.noise_level[0], args.noise_level[1])

            # images in torch are in [c, h, w] format
            batch = batch.permute(0, 3, 1, 2).contiguous().to(device)
            noise = torch.normal(0, noise_level / 255.0, size=batch.size()).to(device)
            noisy_img = batch + noise

            # auto mixed precision forward pass
            with autocast():
                # the network takes noisy images as input 
                # and returns residual (i.e., skip connections)
                residual = model(noisy_img)
                loss = criterion(residual, noise)

            total_loss += loss.item()

            # backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        psnr = test_model(test_set, model, noise=128.0, device=device)[0].mean(axis=1)

        # print some diagnostic information
        print('epoch %d/%d' % (epoch + 1, args.n_epoch))

        print('total training loss %.2f' % total_loss)
        print('test psnr in %.2f, out %.2f' % (psnr[0], psnr[1]))

        print('time elapsed: %s' % str(datetime.timedelta(
            seconds=time.time() - start_time))[:-4])

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

    # setup for training
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, 
                milestones=args.decay_epoch, gamma=args.decay_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    # setup dataset
    islvrc = ISLVRC(args)
    train_set = islvrc.train_set()
    test_set = islvrc.test_set()

    # training dataset
    data_sampler = DSP(train_set, world_size, rank, shuffle=True, seed=args.seed)
    train_set = DataLoader(train_set, batch_size=args.batch_size, 
                shuffle=False, pin_memory=True, sampler=data_sampler)

    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        train_set.sampler.set_epoch(epoch)
        for count, batch in enumerate(train_set):
            optimizer.zero_grad()

            # choose a noise level for the batch
            noise_level = random.randint(args.noise_level[0], args.noise_level[1])

            # images in torch are in [c, h, w] format
            batch = batch.permute(0, 3, 1, 2).contiguous().to(rank)
            noise = torch.normal(0, noise_level / 255.0, size=batch.size()).to(rank)
            noisy_img = batch + noise

            # auto mixed precision forward pass
            with autocast():
                # the network takes noisy images as input 
                # and returns residual (i.e., skip connections)
                residual = model(noisy_img)
                loss = criterion(residual, noise)

            total_loss += loss.item()

            # backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        
        if rank == 0:
            # print some diagnostic information
            print('epoch %d/%d' % (epoch + 1, args.n_epoch))
            
            psnr = test_model(test_set, model, noise=128.0, device=rank)[0].mean(axis=1)
            print('total average loss %.2f' % total_loss / float(count))
            print('test psnr in %.2f, out %.2f' % (psnr[0], psnr[1]))

            print('time elapsed: %s' % str(datetime.timedelta(
                seconds=time.time() - start_time))[:-4])

    # save the parameters of the model
    if rank == 0:
        print('save model parameters')
        torch.save(model.state_dict(), args.save_dir)

    dist.destroy_process_group()