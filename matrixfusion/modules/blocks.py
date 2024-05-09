from collections import OrderedDict

import torch
import torch.nn as nn

from .abstracts import FusableModule


class FactorizedLinear(FusableModule):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.layers = nn.Sequential(
            nn.Linear(in_features, rank),
            nn.Linear(rank, out_features),
        )

    def fuse(self) -> nn.Linear:
        w1 = self.layers[0].weight
        b1 = self.layers[0].bias
        w2 = self.layers[1].weight
        b2 = self.layers[1].bias
        state_dict = OrderedDict([("weight", w2 @ w1), ("bias", w2 @ b1 + b2)])
        fused_layers = nn.Linear(
            self.in_features,
            self.out_features,
            device=state_dict["weight"].device,
            dtype=state_dict["weight"].dtype,
        )
        fused_layers.load_state_dict(state_dict)
        return fused_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TripleLinear(FusableModule):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.main_linear = nn.Linear(in_features, out_features)
        self.factorized_linear = FactorizedLinear(in_features, out_features, rank)

    def fuse(self) -> nn.Linear:
        main_weight = self.main_linear.weight
        main_bias = self.main_linear.bias
        fused_factorized_linear = self.factorized_linear.fuse()
        factorized_weight = fused_factorized_linear.weight
        factorized_bias = fused_factorized_linear.bias
        state_dict = OrderedDict(
            [
                ("weight", main_weight + factorized_weight),
                ("bias", main_bias + factorized_bias),
            ]
        )
        fused_layers = nn.Linear(
            self.in_features,
            self.out_features,
            device=state_dict["weight"].device,
            dtype=state_dict["weight"].dtype,
        )
        fused_layers.load_state_dict(state_dict)
        return fused_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main_linear(x) + self.factorized_linear(x)


class TripleGELU(FusableModule):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.layers = nn.Sequential(
            TripleLinear(in_features, out_features, rank),
            nn.GELU(),
        )

    def fuse(self) -> nn.Sequential:
        return nn.Sequential(self.layers[0].fuse(), self.layers[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
