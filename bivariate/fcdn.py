import torch
import torch.nn as nn

# A fully connected denoiser for bivariate input
class Denoiser(nn.Module):
    def __init__(self, n_node, n_int, bias=False) -> None:
        super().__init__()

        # Setup the network
        inter = []
        for i in range(n_int):
            inter.append(nn.Linear(n_node, n_node, bias=bias))
            inter.append(nn.ReLU())

        self.model = nn.Sequential(
            nn.Linear(2, n_node, bias=bias),
            nn.ReLU(),
            *inter,
            nn.Linear(n_node, 2, bias=bias))

    def forward(self, x):
        return self.model(x)

    def score(self, x):
        with torch.no_grad():
            return - self.model(x)