import torch
import torch.nn as nn

class StackedPoseLifter(nn.Module):

    def __init__(
        self,
        num_joints: int,
        features: int,
        num_layers: int,
        layer_size: int,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.features = features

        self.num_layers = num_layers
        self.layer_size = layer_size

        mlp_layers: list[nn.Module] = [
            nn.Linear(num_joints * features * 2, layer_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)]
        
        for _ in range(num_layers):
            mlp_layers.append(nn.Linear(layer_size, layer_size))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(dropout))

        mlp_layers.append(nn.Linear(layer_size, num_joints * features))

        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_3d_transformer: torch.Tensor, input_3d_FCN: torch.Tensor):
        """
        Forward pass of the model.
        Args:
            input_3d_transformer: Tensor of shape (B, 1, J, features)
            input_3d_FCN: Tensor of shape (B, 1, J, features)
        Returns:
            Tensor of shape (B, 1, J, features)
        """
        input_3d_transformer = input_3d_transformer.view(input_3d_transformer.size(0), -1)
        input_3d_FCN = input_3d_FCN.view(input_3d_FCN.size(0), -1)

        input_combined = torch.cat((input_3d_transformer, input_3d_FCN), dim=-1)

        x = input_combined.view(input_combined.size(0), -1)
        for layer in self.mlp_layers:
            x = layer(x)

        output = x.view(x.size(0), 1, self.num_joints, self.features)
        return output
