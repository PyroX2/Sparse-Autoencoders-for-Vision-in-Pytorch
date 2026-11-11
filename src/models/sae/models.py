import torch
from torch import nn


class BaseSAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _apply_topk(hidden: torch.Tensor, topk: int) -> torch.Tensor:
        batch_size, _ = hidden.shape
        hidden = hidden.flatten(start_dim=0)  # Flatten including batch dimension
        values, indices = torch.topk(hidden, topk * batch_size, dim=0)
        sparse_hidden = torch.zeros_like(hidden)
        sparse_hidden.scatter_(0, indices, values)
        sparse_hidden = sparse_hidden.reshape((batch_size, -1))
        return sparse_hidden


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
        self.weights = nn.Parameter(torch.empty(n_inputs, n_hidden), requires_grad=True)
        nn.init.xavier_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros(n_inputs), requires_grad=True)
        self.hidden_dim = n_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using using encoder layer.

        Args:
                x: Input tensor.
        Returns:
                torch.Tensor: Autoencoder hidden vector.
        """
        return torch.relu(torch.einsum("...i, ij -> ...j", x, self.weights))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using using decoder layer.

        Args:
                hidden: Hidden vector which is an output of the encoder layer.
        Returns:
                torch.Tensor: Reconstructed output.
        """
        return torch.einsum("...j, ij -> ...i", hidden, self.weights) + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.topk = topk
        self.weights = nn.Parameter(torch.empty(n_inputs, n_hidden), requires_grad=True)
        nn.init.xavier_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros(n_inputs), requires_grad=True)
        self.hidden_dim = n_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using using encoder layer.

        Args:
                x: Input tensor.
        Returns:
                torch.Tensor: Autoencoder hidden vector.
        """
        return torch.relu(torch.einsum("...i, ij -> ...j", x, self.weights))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using using decoder layer.

        Args:
                hidden: Hidden vector which is an output of the encoder layer.
        Returns:
                torch.Tensor: Reconstructed output.
        """
        return torch.einsum("...j, ij -> ...i", hidden, self.weights) + self.bias

    def forward(self, x: torch.Tensor, apply_topk: bool = True) -> torch.Tensor:
        """
        Forward function that encodes the input, applies topk and reconstructs it using decoder.

        Args:
                x: Input tensor.

        Returns:
                torch.Tensor: Reconstructed input.
                torch.Tensor: Sparse hidden vector which is an output of the encoder layer.
        """
        hidden = self.encode(x)

        if apply_topk:
            values, indices = torch.topk(hidden, self.topk, dim=1)
            sparse_hidden = torch.zeros_like(hidden)
            sparse_hidden.scatter_(1, indices, values)
        else:
            sparse_hidden = hidden

        output = self.decode(sparse_hidden)
        return output, sparse_hidden


class BatchTopkSAE(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, topk: int) -> None:
        super().__init__()
        self.topk = topk
        self.weights = nn.Parameter(torch.empty(n_inputs, n_hidden), requires_grad=True)
        nn.init.xavier_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros(n_inputs), requires_grad=True)
        self.hidden_dim = n_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using using encoder layer.

        Args:
                x: Input tensor.
        Returns:
                torch.Tensor: Autoencoder hidden vector.
        """
        return torch.relu(torch.einsum("...i, ij -> ...j", x, self.weights))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using using decoder layer.

        Args:
                hidden: Hidden vector which is an output of the encoder layer.
        Returns:
                torch.Tensor: Reconstructed output.
        """
        return torch.einsum("...j, ij -> ...i", hidden, self.weights) + self.bias

    def forward(self, x: torch.Tensor, apply_topk: bool = True) -> torch.Tensor:
        """
        Forward function that encodes the input and reconstructs it using decoder.

        Args:
                x: Input tensor.

        Returns:
                torch.Tensor: Reconstructed input.
                torch.Tensor: Hidden vector which is an output of the encoder layer.
        """
        hidden = self.encode(x)

        if apply_topk:
            batch_size, _ = hidden.shape
            hidden = hidden.flatten(start_dim=0)  # Flatten including batch dimension
            values, indices = torch.topk(hidden, self.topk * batch_size, dim=0)
            sparse_hidden = torch.zeros_like(hidden)
            sparse_hidden.scatter_(0, indices, values)
            sparse_hidden = sparse_hidden.reshape((batch_size, -1))
        else:
            sparse_hidden = hidden

        output = self.decode(sparse_hidden)
        return output, sparse_hidden


class MatryoshkaBatchTopkSAE(BaseSAE):
    def __init__(self, n_inputs: int, expansion_factors, topk: int | list) -> None:
        """
        Matryoshka SAE that applies TopK at batch level.
        Args:
                n_inputs: Number of SAE input features.
                expansion_factors: list of the expansion factors for each SAE inside the Matryoshka.
                    Think of it like creating multiple SAEs with those expansion factors that share weights between each other.
                topk: TopK applied at batch level
        """
        super().__init__()
        self.topk = topk
        self.n_inputs = n_inputs
        self.hidden_dim = int(expansion_factors[-1] * n_inputs)
        self.expansion_factors = torch.tensor(expansion_factors)
        self.weights = nn.Parameter(
            torch.empty(n_inputs, self.hidden_dim, dtype=torch.float32),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros(n_inputs), requires_grad=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input using using encoder layer.

        Args:
                x: Input tensor.
        Returns:
                torch.Tensor: Autoencoder hidden vector.
        """
        return torch.relu(torch.einsum("...i, ij -> ...j", x, self.weights))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input using using decoder layer.

        Args:
                hidden: Hidden vector which is an output of the encoder layer.
        Returns:
                torch.Tensor: Reconstructed output.
        """
        contribution = hidden.unsqueeze(-1) * self.weights.T.unsqueeze(0)
        decoded_cumsum = torch.cumsum(contribution, dim=1)

        # Decoded of shape [batch_size, #expansion_factors, n_inputs]
        decoded = (
            decoded_cumsum[
                :, (self.expansion_factors * self.n_inputs).to(torch.int32) - 1
            ]
            + self.bias
        )

        return decoded

    def forward(self, x: torch.Tensor, apply_topk: bool = True) -> torch.Tensor:
        """
        Forward function that encodes the input and reconstructs it using decoder.

        Args:
                x: Input tensor.

        Returns:
                torch.Tensor: Reconstructed input.
                torch.Tensor: Hidden vector which is an output of the encoder layer.
        """
        hidden = self.encode(x)

        if apply_topk:
            batch_size, _ = hidden.shape
            hidden = hidden.flatten(start_dim=0)  # Flatten including batch dimension
            values, indices = torch.topk(hidden, self.topk * batch_size, dim=0)
            sparse_hidden = torch.zeros_like(hidden)
            sparse_hidden.scatter_(0, indices, values)
            sparse_hidden = sparse_hidden.reshape((batch_size, -1))
        else:
            sparse_hidden = hidden

        # Output of shape [Batch size, # of nested SAEs, input dim]
        output = self.decode(sparse_hidden)
        return output, sparse_hidden
