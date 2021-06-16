import torch, torch.nn as nn, time, datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

def train_denoiser(train_set, model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.train().to(device)
    optimizer = Adam(model.parameters())
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
    criterion = nn.MSELoss()

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size, 
                shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(args.n_epoch):
        print('epoch: %d/%d' % (epoch, args.n_epoch))

        total_loss = 0.0
        start_time = time.time()
        for _, batch in enumerate(train_set):
            optimizer.zero_grad()
            
            # images in torch are in [c, h, w] format
            batch = batch.permute(0, 3, 1, 2).contiguous().to(device)
            noise = torch.normal(0, args.noise_level / 255.0, batch.size()).to(device)
            noisy_img = batch + noise

            # the network takes noisy images as input 
            # and returns residual (i.e., skip connections)
            residual = model(noisy_img)

            loss = criterion(residual, noise)
            loss.backward()
            total_loss += loss.item()

            optimizer.step()

        scheduler.step()

        print('total training loss: %.3f' % total_loss)
        print('time elapsed: %s' % str(datetime.timedelta(
            seconds=time.time() - start_time))[:-4])

    return model.eval().cpu()