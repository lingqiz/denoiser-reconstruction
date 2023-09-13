import torch
import torch.nn as nn

# A fully connected denoiser for bivariate input
class Denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Setup the network
        N_NODE = 16
        N_INT = 5
        inter = []
        for i in range(N_INT):
            inter.append(nn.Linear(N_NODE, N_NODE, bias=False))
            inter.append(nn.ReLU())

        self.model = nn.Sequential(
            nn.Linear(2, N_NODE, bias=False),
            nn.ReLU(),
            *inter,
            nn.Linear(N_NODE, 2, bias=False))

    def forward(self, x):
        return self.model(x)