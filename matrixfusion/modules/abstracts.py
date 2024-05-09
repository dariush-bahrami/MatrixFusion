from abc import ABC, abstractmethod
import torch.nn as nn


class FusableModule(ABC, nn.Module):
    @abstractmethod
    def fuse(self) -> nn.Module:
        pass
