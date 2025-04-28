# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch

class CamLSTMBase(nn.Module):
    """
    Do not instantiate this class.
    Base class for LSTM models that use camera matrices.
    """
    cam_mat_shape = (3, 4)

    def __init__(self, num_joints_in, in_features, num_joints_out, out_features,
                 hidden_size, num_cells, head_layers, dropout=0.25):
        super().__init__()

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.out_features = out_features

        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.head_layers = head_layers

        self.dropout = dropout
    
    def sliding_window(self, inputs_2d, inputs_cam, window_size):
        _, T, J, _ = inputs_2d.shape
        n_windows = T - window_size + 1
        if n_windows <= 0:
            raise ValueError("window_size larger than sequence length")

        axis_permutation = (0, 3, 1, 2)
        win_2d = inputs_2d.unfold(1, window_size, 1).squeeze().permute(axis_permutation)
        win_cam = inputs_cam.unfold(1, window_size, 1).squeeze().permute(axis_permutation)

        out = self(win_2d, win_cam)
        return out.view(1, n_windows, self.num_joints_out, self.out_features)


class CoupledLSTM(CamLSTMBase):
    """
    Basic LSTM model for predicting 3d poses using series of camera matrices
    """

    cam_mat_shape = (3, 4)

    def __init__(self, num_joints_in, in_features, num_joints_out, out_features,
                 hidden_size, num_cells, head_layers, dropout=0.25):
        """ 
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        num_layers -- number of LSTM cells to stack
        hidden_size -- number of features to use in LSTM state
        head_layers -- layer sizes for MLP head
        dropout -- dropout probability
        """
        super().__init__(num_joints_in, in_features, num_joints_out, out_features,
                 hidden_size, num_cells, head_layers, dropout)

        self.lstm_layers = nn.LSTM(num_joints_in*in_features + self.cam_mat_shape[0]*self.cam_mat_shape[1], 
                                   hidden_size, num_cells, batch_first=True, dropout=dropout)

        mlp_layers = [nn.Linear(hidden_size, head_layers[0]), nn.LeakyReLU(), nn.Dropout(dropout)]
        
        for i in range(len(head_layers)-1):
            mlp_layers.append(nn.Linear(head_layers[i], head_layers[i+1]))
            mlp_layers.append(nn.LeakyReLU())
            mlp_layers.append(nn.Dropout(dropout))

        mlp_layers.append(nn.Linear(head_layers[-1], self.out_features*num_joints_out))

        self.mlp_layers = nn.Sequential(*mlp_layers)

    def forward(self, input_2d, input_cam):
        assert len(input_2d.shape) == 4 and len(input_cam.shape) == 4
        assert input_2d.shape[-2] == self.num_joints_in
        assert input_2d.shape[-1] == self.in_features
        
        assert input_cam.shape[-2] == self.cam_mat_shape[0]
        assert input_cam.shape[-1] == self.cam_mat_shape[1]

        # Flatten input (leaving batch and sequence dimension intact)
        flattened_pose_dim = self.num_joints_in*self.in_features
        flattened_input_2d = input_2d.reshape((input_2d.shape[0], input_2d.shape[1], flattened_pose_dim))

        flattened_cam_dim = self.cam_mat_shape[0]*self.cam_mat_shape[1]
        flattened_input_cam = input_cam.reshape((input_cam.shape[0], input_cam.shape[1], flattened_cam_dim))

        x = torch.cat([flattened_input_2d, flattened_input_cam], dim=2)

        # LSTM initial states
        h_0 = torch.zeros(self.num_cells, x.shape[0], self.hidden_size)
        c_0 = torch.zeros(self.num_cells, x.shape[0], self.hidden_size)

        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        lstm_out, _ = self.lstm_layers(x, (h_0, c_0))
        mlp_out = self.mlp_layers(lstm_out[:, -1, :])

        return mlp_out.reshape(input_2d.shape[0], 1, self.num_joints_out, self.out_features)


class UncoupledLSTM(CamLSTMBase):
    cam_mat_shape = (3, 4)

    def __init__(self, num_joints_in, in_features, out_features, num_joints_out,
                 hidden_size, num_cells, head_layers, dropout=0.25):
        """ 
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        out_features -- number of output features for each joint (typically 3 for 3D output)
        num_layers -- number of LSTM cells to stack
        hidden_size -- number of features to use in LSTM state
        head_layers -- layer sizes for MLP head
        dropout -- dropout probability
        """
        super().__init__(num_joints_in, in_features, num_joints_out, out_features,
            hidden_size, num_cells, head_layers)

        self.lstm_layers = nn.LSTM(num_joints_in*in_features + self.cam_mat_shape[0]*self.cam_mat_shape[1], 
                                   hidden_size, num_cells, batch_first=True, dropout=dropout)

        mlp_layers = [nn.Linear(hidden_size, head_layers[0]), nn.LeakyReLU(), nn.Dropout(dropout)]
        
        for i in range(len(head_layers)-1):
            mlp_layers.append(nn.Linear(head_layers[i], head_layers[i+1]))
            mlp_layers.append(nn.LeakyReLU())
            mlp_layers.append(nn.Dropout(dropout))

        mlp_layers.append(nn.Linear(head_layers[-1], 2))

        self.mlp_layers = nn.Sequential(*mlp_layers)

    def forward(self, input_2d: torch.Tensor, input_cam: torch.Tensor):
        assert input_2d.ndim  == 4 and input_cam.ndim == 4
        B, T, J, F = input_2d.shape
        assert J == self.num_joints_in  and F == self.in_features
        H, W = self.cam_mat_shape
        assert input_cam.shape[-2:] == (H, W)

        flat_pose = input_2d.view(B, T, self.num_joints_in * self.in_features)
        flat_cam  = input_cam.view(B, T, H * W)
        x = torch.cat([flat_pose, flat_cam], dim=-1)

        h_0 = torch.zeros(self.num_cells, B, self.hidden_size)
        c_0 = torch.zeros(self.num_cells, B, self.hidden_size)

        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        lstm_out, _ = self.lstm_layers(x, (h_0, c_0))

        lambdas = torch.sigmoid(self.mlp_layers(lstm_out))
        lambda_pose, lambda_cam = lambdas[..., 0], lambdas[..., 1]

        ref_pose = self.compute_filtered(input_2d, lambda_pose)
        ref_cam  = self.compute_filtered(input_cam, lambda_cam)

        ref_pose = ref_pose.view(B, self.num_joints_in, self.in_features)
        ref_cam  = ref_cam.view(B, self.cam_mat_shape[0], self.cam_mat_shape[1])

        # Triangulate points
        triangulated_poses = self.triangulate_points_batched(input_cam[:, 0, :, :], input_cam[:, T//2, :, :],
                                                             input_2d[:, 0, :, :], input_2d[:, T//2, :, :])
        return triangulated_poses.reshape(B, 1, self.num_joints_out, self.out_features)

    def compute_filtered(self, ref_seq, lambda_seq):
        rev = torch.flip(lambda_seq, dims=[1])
        cumprod_rev = torch.cumprod(rev, dim=1)
        cumprod = torch.flip(cumprod_rev, dims=[1])

        prod_next = torch.cat([cumprod[:, 1:],
                               torch.ones(lambda_seq.shape[0], 1, device=lambda_seq.device)], dim=1)
        if torch.cuda.is_available():
            prod_next = prod_next.cuda()
        w = (1.0 - lambda_seq) * prod_next

        while w.ndim < ref_seq.ndim:
            w = w.unsqueeze(-1)
        return (ref_seq * w).sum(dim=1)

    def triangulate_points_batched(self, cam_mat_a, cam_mat_b, screen_a, screen_b):
        B, N = screen_a.shape[0], screen_a.shape[1]

        hom_screen_a = torch.cat([screen_a, torch.ones(B, N, 1, device=screen_a.device, dtype=screen_a.dtype)], dim=-1)
        hom_screen_b = torch.cat([screen_b, torch.ones(B, N, 1, device=screen_b.device, dtype=cam_mat_b.dtype)], dim=-1)

        hom_screen_a = hom_screen_a.reshape(B * N, self.out_features)
        hom_screen_b = hom_screen_b.reshape(B * N, self.out_features)

        # Expand camera matrices to match screen coordinates
        cam_mat_a = cam_mat_a.unsqueeze(1).expand(B, N, self.cam_mat_shape[0], self.cam_mat_shape[1])
        cam_mat_a = cam_mat_a.reshape(B * N, self.cam_mat_shape[0], self.cam_mat_shape[1])

        cam_mat_b = cam_mat_b.unsqueeze(1).expand(B, N, self.cam_mat_shape[0], self.cam_mat_shape[1])
        cam_mat_b = cam_mat_b.reshape(B * N, self.cam_mat_shape[0], self.cam_mat_shape[1])

        # Build SVD matrix using both batches of observations (see standard epipolar triangulation)
        A = torch.zeros((B * N, 4, 4), device=screen_a.device, dtype=screen_a.dtype)
        A[:, 0, :] = hom_screen_a[:, 0:1] * cam_mat_a[:, 2, :] - cam_mat_a[:, 0, :]
        A[:, 1, :] = hom_screen_a[:, 1:2] * cam_mat_a[:, 2, :] - cam_mat_a[:, 1, :]

        A[:, 2, :] = hom_screen_b[:, 0:1] * cam_mat_b[:, 2, :] - cam_mat_b[:, 0, :]
        A[:, 3, :] = hom_screen_b[:, 1:2] * cam_mat_b[:, 2, :] - cam_mat_b[:, 1, :]

        # 4. solve
        _, _, Vh = torch.linalg.svd(A, full_matrices=False)
        hom = Vh[:, -1]                                   # (B*N, 4)

        # 5. normalise robustly
        scale = hom.abs().amax(dim=1, keepdim=True)
        hom = hom / scale                                 # now max component ≈1
        w = hom[:, 3:4]

        # optional masking
        bad = w.abs() < 0.005
        w_safe = torch.where(bad, torch.ones_like(w), w)

        xyz = hom[:, :3] / w_safe
        xyz[bad.expand_as(xyz)] = float('nan')            # mark invalid

        return xyz.reshape(B, N, 3)
