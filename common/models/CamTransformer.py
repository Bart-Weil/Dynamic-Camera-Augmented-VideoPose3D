import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in the original Transformer.
    Args:
        d_model: embedding dimension
        dropout: dropout probability
        max_len: maximum sequence length supported
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        Args:
            x: Tensor of shape (B, T, d_model)
        Returns:
            Tensor of same shape as x with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CamTransformerBase(nn.Module):
    """
    Base class for Transformer models that use camera matrices.
    Do not instantiate directly; use a subclass (e.g. CoupledTransformer).
    """

    cam_mat_shape = (3, 4)

    def __init__(
        self,
        num_joints_in: int,
        in_features: int,
        num_joints_out: int,
        out_features: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        head_layers: list[int],
        dropout: float = 0.25,
    ):
        super().__init__()

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.out_features = out_features

        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.head_layers = head_layers
        self.dropout = dropout

    def sliding_window(self, inputs_2d: torch.Tensor, inputs_cam: torch.Tensor, window_size: int):
        """Apply model over a sliding window and stack outputs.
        Args:
            inputs_2d: (B=1, T, J, in_features)
            inputs_cam: (B=1, T, 3, 4)
            window_size: size of temporal window
        Returns:
            Tensor of shape (1, n_windows, num_joints_out, out_features)
        """
        _, T, J, _ = inputs_2d.shape
        n_windows = T - window_size + 1
        if n_windows <= 0:
            raise ValueError("window_size larger than sequence length")

        axis_permutation = (0, 3, 1, 2)  # Move feature dim before time
        win_2d = inputs_2d.unfold(1, window_size, 1).squeeze().permute(axis_permutation)
        win_cam = inputs_cam.unfold(1, window_size, 1).squeeze().permute(axis_permutation)

        out = self(win_2d, win_cam)
        return out.view(1, n_windows, self.num_joints_out, self.out_features)


class CoupledTransformer(CamTransformerBase):
    """Transformer model for predicting 3D poses using series of camera matrices:
        num_joints_in: Number of input joints per frame (e.g., 17 for Human3.6M).
        in_features: Number of features per joint (typically 2 for 2D poses).
        num_joints_out: Number of output joints (can differ from input).
        out_features: Number of features per output joint (typically 3 for 3D poses).
        d_model: Dimensionality of transformer embeddings.
        num_layers: Number of transformer encoder layers.
        nhead: Number of attention heads in each encoder layer.
        dim_feedforward: Dimension of the feedforward network in each encoder layer.
        head_layers: List of hidden layer sizes for the output MLP head.
        dropout: Dropout probability for transformer and MLP layers.
    """
    def __init__(
        self,
        num_joints_in: int,
        in_features: int,
        num_joints_out: int,
        out_features: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        head_layers: list[int],
        dropout: float = 0.25,
    ):
        super().__init__(
            num_joints_in,
            in_features,
            num_joints_out,
            out_features,
            d_model,
            num_layers,
            nhead,
            dim_feedforward,
            head_layers,
            dropout,
        )

        # Embedding layer to project concatenated inputs to d_model
        concat_dim = num_joints_in * in_features + self.cam_mat_shape[0] * self.cam_mat_shape[1]
        self.input_projection = nn.Linear(concat_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head (same pattern as LSTM version)
        mlp_layers: list[nn.Module] = [
            nn.Linear(d_model, head_layers[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        ]
        for i in range(len(head_layers) - 1):
            mlp_layers.extend(
                [
                    nn.Linear(head_layers[i], head_layers[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                ]
            )
        mlp_layers.append(nn.Linear(head_layers[-1], out_features * num_joints_out))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def forward(self, input_2d: torch.Tensor, input_cam: torch.Tensor):
        """Forward pass.
        Args:
            input_2d: Tensor (B, T, J_in, in_features)
            input_cam: Tensor (B, T, 3, 4)
        Returns:
            Tensor (B, 1, num_joints_out, out_features)
        """
        # Shape validations
        assert len(input_2d.shape) == 4 and len(input_cam.shape) == 4, "Invalid input dims"
        assert (
            input_2d.shape[-2] == self.num_joints_in and input_2d.shape[-1] == self.in_features
        ), "Unexpected 2D input shape"
        assert (
            input_cam.shape[-2] == self.cam_mat_shape[0] and input_cam.shape[-1] == self.cam_mat_shape[1]
        ), "Unexpected camera matrix shape"

        B, T, _, _ = input_2d.shape

        # Flatten inputs
        flat_2d = input_2d.reshape(B, T, -1)  # (B, T, J_in * in_features)
        flat_cam = input_cam.reshape(B, T, -1)  # (B, T, 12)
        x = torch.cat([flat_2d, flat_cam], dim=2)  # (B, T, concat_dim)

        # Project to d_model and add positional encoding
        x = self.input_projection(x)  # (B, T, d_model)
        x = self.positional_encoding(x)  # (B, T, d_model)

        # Transformer encoder
        enc_out = self.transformer_encoder(x)  # (B, T, d_model)

        # Use representation of last time step
        last_feat = enc_out[:, -1, :]  # (B, d_model)

        # MLP head and reshape output
        mlp_out = self.mlp_layers(last_feat)  # (B, num_joints_out * out_features)
        return mlp_out.view(B, 1, self.num_joints_out, self.out_features)
