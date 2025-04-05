# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch

class CoupledLSTM(nn.Module):
    """
    Basic LSTM model for predicting 3d poses using series of camera matrices
    """

    cam_mat_shape = (3, 4)

    def __init__(self, num_joints_in, in_features, num_joints_out,
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
        super().__init__()

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out

        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.head_layers = head_layers

        self.lstm_layers = nn.LSTM(num_joints_in*in_features + self.cam_mat_shape[0]*self.cam_mat_shape[1], 
                                   hidden_size, num_cells, batch_first=True, dropout=dropout)

        mlp_layers = [nn.Linear(hidden_size, head_layers[0])]
        
        for i in range(len(head_layers)-1):
            mlp_layers.append(nn.Linear(head_layers[i], head_layers[i+1]))
            mlp_layers.append(nn.LeakyReLU())
            mlp_layers.append(nn.Dropout(dropout))

        mlp_layers.append(nn.Linear(head_layers[-1], num_joints_out*3))

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
        c_0 = torch.zeros(self.num_cells, x.shape[0], self.hidden_size)
        h_0 = torch.zeros(self.num_cells, x.shape[0], self.hidden_size)

        if torch.cuda.is_available():
            c_0 = c_0.cuda()
            h_0 = h_0.cuda()

        lstm_out, _ = self.lstm_layers(x, (h_0, c_0))
        mlp_out = self.mlp_layers(lstm_out[:, -1, :])

        return mlp_out
