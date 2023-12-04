"""Module containing different components of multi-head attention heads."""

import torch
import torch.nn.functional as F

# todo: add named tensors


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        """Initialize MLP.

        Args:
            input_dim: Dimensionality of the input features.
            hidden_dim: Number of units in the hidden layers.
            output_dim: Dimensionality of the output features.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        if self.num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(n, k)
                    for n, k in zip([input_dim] + h, h + [output_dim])
                ]
            )
            if self.dropout > 0.0:
                self.dropouts = torch.nn.ModuleList(
                    [torch.nn.Dropout(dropout) for _ in range(self.num_layers - 1)]
                )
        else:
            self.layers = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor of shape (batch_size, num_instances, input_dim).

        Returns:
            Output tensor of shape (batch_size, num_instances, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1 and self.dropout > 0.0:
                x = self.dropouts[i](x)

        return x


class ATTWeightHead(torch.nn.Module):
    """Single attention head."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """Initialize an instance of ATTWeightHead.

        Args:
            feature_dim: The dimensionality of input features.
            num_layers: The number of hidden layers in the MLP.
            dropout: Dropout probability.
        """
        super().__init__()

        self.q_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)
        self.k_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the attention weights of a query tensor using the key tensor.

        Args:
            query: Input tensor of shape (batch_size, num_frame_instances, feature_dim).
            key: Input tensor of shape (batch_size, num_window_instances, feature_dim).

        Returns:
            Output tensor of shape (batch_size, num_frame_instances, num_window_instances).
        """
        k = self.k_proj(key)
        q = self.q_proj(query)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        return attn_weights  # (B, N_t, N)
