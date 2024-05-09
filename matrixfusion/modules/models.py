import torch
import torch.nn as nn

from .abstracts import FusableModule
from .blocks import TripleGELU, TripleLinear


class TripleMLP(FusableModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        rank: int,
    ):
        super().__init__()
        layers = []
        layers.append(TripleGELU(in_features, hidden_features, rank))
        for _ in range(hidden_layers):
            layers.append(TripleGELU(hidden_features, hidden_features, rank))
        layers.append(TripleLinear(hidden_features, out_features, rank))
        self.layers = nn.Sequential(*layers)

    def fuse(self) -> nn.Sequential:
        return nn.Sequential(*[layer.fuse() for layer in self.layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
