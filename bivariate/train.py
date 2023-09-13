import torch
from torch.utils.data import DataLoader

def train_simple(train_set, test_set, model):
    # training with GPU if available
    rank = (0 if torch.cuda.is_available() else 'cpu')
    model = model.to(rank)

    # training dataset
    train_set = DataLoader(train_set, batch_size=args.batch_size,
                shuffle=True, num_workers=4, pin_memory=True)