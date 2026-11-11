import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Sequential()
        self.fc.add_module("flatten", nn.Flatten())
        self.fc.add_module("linear_01", nn.Linear(3 * 784, 512))
        self.fc.add_module("tanh_01", nn.Tanh())
        self.fc.add_module("linear_02", nn.Linear(512, 64))
        self.fc.add_module("tanh_02", nn.Tanh())
        self.fc.add_module("linear_03", nn.Linear(64, 8))
        self.fc.add_module("tanh_03", nn.Tanh())
        self.fc.add_module("output", nn.Linear(8, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("CNN model class is not implemented.")

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ViT model class is not implemented.")

