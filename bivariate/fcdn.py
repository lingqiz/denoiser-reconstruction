import torch
import torch.nn as nn

# A fully connected denoiser for bivariate input
class Denoiser(nn.Module):
    def __init__(self, n_node, n_int) -> None:
        super().__init__()

        # Setup the network
        inter = []
        for i in range(n_int):
            inter.append(nn.Linear(n_node, n_node, bias=False))
            inter.append(nn.ReLU())

        self.model = nn.Sequential(
            nn.Linear(2, n_node, bias=False),
            nn.ReLU(),
            *inter,
            nn.Linear(n_node, 2, bias=False))

    def forward(self, x):
        return self.model(x)