import torch, time, datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
from utils.training import sample_noise
from dataclasses import dataclass

@dataclass
class Args:
    noise_level: list
    n_epoch: int = 50
    batch_size: int = 256
    lr: float = 0.01
    decay_lr: float = 0.99
    biased: bool = True

def train_simple(train_set, test_set, model, args):
    # training with GPU if available
    rank = (0 if torch.cuda.is_available() else 'cpu')
    model = model.to(rank)
    model.train()

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size,
                shuffle=True, num_workers=4, pin_memory=True)

    # optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.90)
    scheduler = ExponentialLR(optimizer, gamma=args.decay_lr)
    criterion = nn.MSELoss()

    # run training
    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for count, batch in enumerate(train_set):
            optimizer.zero_grad(set_to_none=True)

            # setup noise and input pair
            batch = batch.to(rank)
            noise = sample_noise(batch.shape, args.noise_level, args.biased).to(rank)
            noise_input = batch + noise

            # forward pass
            residual = model(noise_input)
            loss = criterion(residual, noise)

            # backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # print some diagnostic information
        print('epoch %d/%d' % (epoch + 1, args.n_epoch))
        print('time elapsed: %s' % str(datetime.timedelta(
                seconds=time.time() - start_time))[:-4])
        print('average loss %.6f' % (total_loss / float(count + 1)))

        # performance on test set
        model.eval()

        test_noise = test_set + torch.normal(0, 0.5, size=test_set.size())
        with torch.no_grad():
            residual = model(test_noise.to(rank))
            test_denoise = test_noise - residual.cpu()

            test_in = criterion(test_set, test_noise)
            test_out = criterion(test_set, test_denoise)

        print('test in %.4f, out %.4f' % (test_in, test_out))