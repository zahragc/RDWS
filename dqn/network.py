import torch
import torch.nn as nn
from torch import Tensor

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device):
        """Initialization."""
        super(Network, self).__init__()
        dim = 256

        # def initWeights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform(m.weight)
        #         m.bias.data.fill_(0.01)


        # 256 128 256 BAD
        # 128 128 128 BAD

        self.layers = nn.Sequential(
            nn.Linear(in_dim, dim),

            nn.Linear(dim, dim),
            # nn.LayerNorm(dim),
            # nn.Dropout(p=0.2),
            nn.Tanh(), #ReLU

            nn.Linear(dim, dim), 
            # nn.LayerNorm(dim),
            # nn.Dropout(p=0.2),
            nn.Tanh(), #ReLU

            nn.Linear(dim, out_dim),
        )

        # self.layers.apply(initWeights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x);