import torch
from torch import nn
from typing import List


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


class TrimmedClsModel(nn.Module):
    def __init__(
        self, cls_model: nn.Module, sae_model: nn.Module, neurons_to_kill: List = []
    ):
        super().__init__()
        self.cls_model = cls_model
        self.cls_backbone = self.cls_model.fc[:-3]
        self.cls_head = self.cls_model.fc[-3:]

        self.sae_model = sae_model
        self.neurons_to_kill = neurons_to_kill

    def forward(self, x):
        # Embedding extraction using cls backbone
        embedding = self.cls_backbone(x)

        # SAE embedding modification
        hidden = self.sae_model.encode(embedding)
        modified_hidden = hidden
        modified_hidden[:, self.neurons_to_kill] = 0

        # values, indices = torch.topk(modified_hidden, self.sae_model.topk, dim=1)
        # sparse_hidden = torch.zeros_like(modified_hidden)
        # sparse_hidden.scatter_(1, indices, values)

        sae_output = self.sae_model.decode(modified_hidden)

        cls_output = self.cls_head(sae_output)
        return cls_output


class SAE(nn.Module):
    """Sparse autoencoder built from simple Linear layers."""

    def __init__(self, n_inputs: int, n_hidden: int) -> None:
        """
        Initialize function.

        Args:
                n_inputs: Shape of input layer.
                n_hidden: Shape of hidden vector.
        """
        super().__init__()
        self.encoder = nn.Linear(n_inputs, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_inputs)
        self.decoder.weight = torch.nn.Parameter(self.encoder.weight.T)
        self.hidden_dim = n_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using using encoder layer.

        Args:
                x: Input tensor.
        Returns:
                torch.Tensor: Autoencoder hidden vector.
        """
        return torch.relu(self.encoder(x))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using using decoder layer.

        Args:
                hidden: Hidden vector which is an output of the encoder layer.
        Returns:
                torch.Tensor: Reconstructed output.
        """
        return self.decoder(hidden)

    def forward(self, x):
        """
        Forward function that encodes the input and reconstructs it using decoder.

        Args:
                x: Input tensor.

        Returns:
                torch.Tensor: Reconstructed input.
                torch.Tensor: Hidden vector which is an output of the encoder layer.
        """
        hidden = self.encode(x)
        output = self.decode(hidden)
        return output, hidden


class TopkSAE(nn.Module):
    """Sparse autoencoder built from simple Linear layers."""

    def __init__(self, n_inputs: int, n_hidden: int, topk: int) -> None:
        """
        Initialize function.

        Args:
                n_inputs: Shape of input layer.
                n_hidden: Shape of hidden vector.
                topk: Number of top values to keep.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(), nn.Linear(n_hidden, n_inputs)
        )
        self.topk = topk
        self.hidden_dim = n_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using using encoder layer. Doesn't apply topk.

        Args:
                x: Input tensor.
        Returns:
                torch.Tensor: Autoencoder hidden vector.
        """
        return self.encoder(x)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using using decoder layer.

        Args:
                hidden: Hidden vector which is an output of the encoder layer.
        Returns:
                torch.Tensor: Reconstructed output.
        """
        return self.decoder(hidden)

    def forward(self, x):
        """
        Forward function that encodes the input, applies topk and reconstructs it using decoder.

        Args:
                x: Input tensor.

        Returns:
                torch.Tensor: Reconstructed input.
                torch.Tensor: Sparse hidden vector which is an output of the encoder layer.
        """
        hidden = self.encode(x)

        values, indices = torch.topk(hidden, self.topk, dim=1)
        sparse_hidden = torch.zeros_like(hidden)
        sparse_hidden.scatter_(1, indices, values)

        output = self.decode(sparse_hidden)
        return output, sparse_hidden
