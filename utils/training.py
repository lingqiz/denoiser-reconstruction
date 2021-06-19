import torch, torch.nn as nn, time, datetime, random
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils.dataset import test_model

def train_denoiser(train_set, test_set, model, args):
    # training with GPU(s) if available
    # use DataParallel for multi-GPU training 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print('Use', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)
        
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