import torch, torch.nn as nn, time, datetime, random, math
import torch.distributed as dist
from models.denoiser import Denoiser
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils.dataset import test_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DSP

def sample_noise(size, noise_level, biased=False):
    noise = torch.empty(size=size)

    for idx in range(int(size[0])):
        # determine noise S.D.
        noise_sd = random.randint(noise_level[0], noise_level[1]) / 255.0
        if biased:
            noise_sd = math.pow(noise_sd, 2)

        # sample Gaussian i.i.d. noise
        noise[idx] = torch.normal(mean=0.0, std=noise_sd, size=list(size[1:]))

    return noise

def train_run(model, train_set, test_set, sampler, rank, args):
    # setup for training
    if args.opt_index == 0:
        # Adam optimizer
        optimizer = Adam(model.parameters(), lr=args.lr)
        
    elif args.opt_index == 1:
        # SGD with momentum
        optimizer = SGD(model.parameters(), lr=args.lr, 
                        momentum=0.90)
        
    elif args.opt_index == 2:
        # Adam with weight decay
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                          weight_decay=args.decay_adam)
    else:
        # invalid option
        optimizer = None        
        
    scheduler = ExponentialLR(optimizer, gamma=args.decay_lr)
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

            # input size: [n, h, w, c]
            # images in torch are in [n, c, h, w] format
            if args.scale_image:
                batch = batch.permute(1, 2, 3, 0)
                batch = batch * torch.rand(batch.size(-1))
                batch = batch.permute(3, 2, 0, 1).contiguous().to(rank)
            else:
                batch = batch.permute(0, 3, 1, 2).contiguous().to(rank)

            noise = sample_noise(batch.size(), args.noise_level, args.bias_sd).to(rank)
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

            if args.data_path == 'intrinsic':
                noise_level = 512.0
                data_range = args.data_range
                clip_range = (args.range_lb, args.range_ub)
            else:
                noise_level = 128.0
                data_range = None
                clip_range = (0, 1)

            psnr = test_model(test_set, model, noise=noise_level, device=rank,
                    data_range=data_range, clip_range=clip_range)[0].mean(axis=1)

            print('average loss %.6f' % (total_loss / float(count + 1)))
            print('test psnr in %.2f, out %.2f' % (psnr[0], psnr[1]))

            print('time elapsed: %s' % str(datetime.timedelta(
                seconds=time.time() - start_time))[:-4])

    # return test performance from the main thread
    if rank == 0 or rank == 'cpu':
        return psnr[0], psnr[1]        

def train_denoiser(train_set, test_set, model, args):
    # training with GPU if available
    rank = (0 if torch.cuda.is_available() else 'cpu')
    model = model.to(rank)

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size,
                shuffle=True, num_workers=4, pin_memory=True)

    train_run(model, train_set, test_set, sampler=False, rank=rank, args=args)
    return model.eval().cpu()

# training with Distributed Data Parallel (process level parallelism)
def train_parallel(rank, world_size, train_set, test_set, args):
    dist.init_process_group("nccl", init_method='env://',
                            rank=rank, world_size=world_size)

    # wrap the model with DDP
    # In DDP, the constructor, the forward pass,
    # and the backward pass are distributed synchronization points
    model = Denoiser(args)
    
    # load model parameters if continue training
    if args.cont_train and rank == 0:
        print('load model parameters from %s' % args.save_path)
        model.load_state_dict(torch.load(args.save_path))

    model = DDP(model.to(rank), device_ids=[rank])

    # load training dataset
    data_sampler = DSP(train_set, world_size, rank, shuffle=True, seed=args.seed)
    train_set = DataLoader(train_set, batch_size=args.batch_size, drop_last=True,
                shuffle=False, num_workers=4, pin_memory=True, sampler=data_sampler)

    psnr = train_run(model, train_set, test_set, sampler=True, rank=rank, args=args)
    
    if rank == 0:
        # save the parameters of the model
        if args.save_model:
            print('save model parameters')
            torch.save(model.module.state_dict(), args.save_path)
            
        # write test performance to file
        else:
            with open('grid_search.log', 'a') as fl:
                 text = 'test psnr in %.2f, out %.2f \n' % (psnr[0], psnr[1])
                 fl.write(text)

    # end process group
    dist.destroy_process_group()