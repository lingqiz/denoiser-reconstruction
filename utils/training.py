from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

def train_denoiser(train_set, model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.train().to(device)
    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.n_epoch):
        print('epoch %d/%d' % (epoch, args.n_epoch))

        for _, batch in enumerate(train_set, 0):
            optimizer.zero_grad()
            
            # images in torch are in [c, h, w] format
            batch = batch.permute(0, 3, 1, 2).contiguous()
            noise = torch.normal(0, args.noise_level/255.0, batch.size()) 
            noisy_img = batch + noise

            noisy_img.requires_grad = True
            noise.requires_grad = True

            noisy_img = noisy_img.to(device)
            noise = noise.to(device)

            # the network takes noisy images as input and returns residual
            residual = model(noisy_img)
            loss = criterion(residual, noise)
            loss.backward()
            optimizer.step()