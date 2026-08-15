from torch import nn


class ClsModel(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 784, 512),
            nn.Tanh(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 8),
            nn.Tanh(),
            nn.Linear(8, n_classes),
        )

    def forward(self, x):
        return self.fc(x)
