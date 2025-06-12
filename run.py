# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import sys
import errno

from common.camera import *
from common.models.TemporalModel import *
from common.models.CamLSTM import *
from common.models.CamTransformer import *
from common.models.StackedPoseLifter import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

from common.datasets.h36m_dataset import Human36mDataset
from common.datasets.humaneva_dataset import HumanEvaDataset
from common.datasets.CMUMocapDataset import CMUMocapDataset
from common.datasets.ThreeDPWDataset import ThreeDPWDataset
from common.custom_dataset import CustomDataset

args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = '/vol/bitbucket/bw1222/data/npz/data_3d_' + args.dataset + '.npz'
match args.dataset:
    case 'h36m':
        dataset = Human36mDataset(dataset_path)
    case 'humaneva':
        dataset = HumanEvaDataset(dataset_path)
    case 'CMU':
        dataset = CMUMocapDataset(dataset_path)
    case 'CMU_3DPW':
        dataset = CMUMocapDataset(dataset_path, use_3DPW=True)
    case '3DPW':
        dataset = ThreeDPWDataset(dataset_path)
    case 'custom':
        dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    case _:
        raise ValueError(f"Unknown dataset: {args.dataset}")

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        
        if 'positions' in anim:
            if isinstance(dataset, (ThreeDPWDataset, CMUMocapDataset)):
                pos_3d = anim['positions']
                pos_3d -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                anim['positions_3d'] = [pos_3d]
            else:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('/vol/bitbucket/bw1222/data/npz/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton_2d().joints_left()), list(dataset.skeleton_2d().joints_right())
keypoints = keypoints['positions_2d'].item()
print(keypoints.keys(), '/vol/bitbucket/bw1222/data/npz/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue
            
        if not isinstance(dataset, (ThreeDPWDataset, CMUMocapDataset)):
            for cam_idx in range(len(keypoints[subject][action])):
                
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
        
for subject in keypoints.keys():
    for action in keypoints[subject]:
        if isinstance(dataset, (ThreeDPWDataset, CMUMocapDataset)):
            kps = keypoints[subject][action]
            intrinsics = dataset.cameras()[subject][action]['intrinsics']
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=intrinsics['res_w'], h=intrinsics['res_h'])
            keypoints[subject][action] = [kps]
        else:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps
           
def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    print(keypoints.keys())
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            if subject in dataset.cameras():
                if isinstance(dataset, (ThreeDPWDataset, CMUMocapDataset)):
                    out_camera_params.append(dataset.cameras()[subject][action])
                else:
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)
    
rng = np.random.default_rng(seed=42)
split_subjects = list(dataset.subjects())
rng.shuffle(split_subjects)

subjects_train, subjects_validate, subjects_test = None, None, None

if not args.evaluate and not args.tune_hyperparameters:
    if args.subjects_train and args.subjects_test:
        subjects_train = args.subjects_train.split(',')
        subjects_test = args.subjects_test.split(',') if args.subjects_test else None
    else:
        # Split subjects into train, validation and test sets
        num_train = int(len(split_subjects) * 0.9)

        subjects_train = split_subjects[:num_train]
        subjects_test = split_subjects[num_train:]
elif args.tune_hyperparameters:
    if args.subjects_train and args.subjects_test and args.subjects_validate:
        subjects_train = args.subjects_train.split(',')
        subjects_validate = args.subjects_validate.split(',')
        subjects_test = args.subjects_test.split(',')
    else:
        num_train = int(len(split_subjects) * 0.8)
        num_validate = int(len(split_subjects) * 0.1)
        num_test = len(split_subjects) - num_train - num_validate
        subjects_train = split_subjects[:num_train]
        subjects_validate = split_subjects[num_train:num_train + num_validate]
        subjects_test = split_subjects[num_train + num_validate:num_train + num_validate + num_test]

elif args.evaluate:
    if args.subjects_test:
        subjects_test = args.subjects_test.split(',') if args.subjects_test != "*" else list(dataset.subjects())
    elif args.viz_subject:
        subjects_test = [args.viz_subject]
    else:
        raise ValueError('Please specify subjects to evaluate on with --subjects-test')

elif args.render:
    subjects_test = [args.viz_subject]
    
else:
    raise ValueError('Invalid mode, please specify --evaluate or --render')

if subjects_train:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)
if subjects_test:
    cameras_test, poses_test, poses_test_2d = fetch(subjects_test, action_filter)
if subjects_validate:
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_validate, action_filter)

print("Subjects: ", dataset.subjects())

match args.model_name:
    case 'StackedPoselifter':
        receptive_field = 243
        pad = (receptive_field - 1) // 2 # Padding on each side
        if args.causal:
            print('INFO: Using causal convolutions')
            causal_shift = pad
        else:
            causal_shift = 0

        # Parse the transformer head architecture
        transformer_head_layers = [int(x) for x in args.transformer_head_architecture.split(',')]

        transformer_hyperparams = {
            'num_joints_in': poses_test_2d[0].shape[-2],
            'in_features': poses_test_2d[0].shape[-1],
            'num_joints_out': poses_test[0].shape[-2],
            'out_features': poses_test[0].shape[-1],
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'n_heads': args.n_heads,
            'dim_feedforward': args.dim_feedforward,
            'head_layers': transformer_head_layers,
            'dropout': args.transformer_dropout
        }

        filter_widths = [int(x) for x in args.fcn_architecture.split(',')]

        model_fcn = TemporalModel(poses_test_2d[0].shape[-2], poses_test_2d[0].shape[-1], dataset.skeleton_3d().num_joints(),
                                        filter_widths=filter_widths, causal=args.causal, dropout=args.fcn_dropout, channels=args.channels,
                                        dense=args.dense)
        
        model_transformer = CoupledTransformer(**transformer_hyperparams)

        model_pos_train = StackedPoseLifter(dataset.skeleton_3d().num_joints(), poses_test[0].shape[-1], args.stacked_num_layers, args.layer_size,
                                            args.stacked_pose_lifter_dropout)
        
        model_pos = StackedPoseLifter(dataset.skeleton_3d().num_joints(), poses_test[0].shape[-1], args.stacked_num_layers, args.layer_size,
                                            args.stacked_pose_lifter_dropout)
        
        print('Loading transformer checkpoint from:', args.transformer_weights)
        transformer_checkpoint = torch.load(args.transformer_weights, map_location=lambda storage, loc: storage, weights_only=False)
        print('Transformer was trained for {} epochs'.format(transformer_checkpoint['epoch']))
        model_transformer.load_state_dict(transformer_checkpoint['model_pos'])

        print('Loading FCN checkpoint from:', args.fcn_weights)
        fcn_checkpoint = torch.load(args.fcn_weights, map_location=lambda storage, loc: storage, weights_only=False)
        print('This model was trained for {} epochs'.format(transformer_checkpoint['epoch']))
        model_fcn.load_state_dict(fcn_checkpoint['model_pos'])

    case 'FCN':
        filter_widths = [int(x) for x in args.fcn_architecture.split(',')]

        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
        model_pos_train = TemporalModel(poses_test_2d[0].shape[-2], poses_test_2d[0].shape[-1], dataset.skeleton_3d().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.fcn_dropout, channels=args.channels,
                                    dense=args.dense)
            
        model_pos = TemporalModel(poses_test_2d[0].shape[-2], poses_test_2d[0].shape[-1], dataset.skeleton_3d().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.fcn_dropout, channels=args.channels,
                                    dense=args.dense)

        receptive_field = model_pos.receptive_field()
        print('INFO: Receptive field: {} frames'.format(receptive_field))
        pad = (receptive_field - 1) // 2 # Padding on each side
        if args.causal:
            print('INFO: Using causal convolutions')
            causal_shift = pad
        else:
            causal_shift = 0

    case 'LSTM-Coupled':
        receptive_field = 243
        pad = (receptive_field - 1) // 2 # Padding on each side
        if args.causal:
            print('INFO: Using causal convolutions')
            causal_shift = pad
        else:
            causal_shift = 0
        lstm_head_layers = [int(x) for x in args.lstm_head_architecture.split(',')]
        
        lstm_hyperparams = {
            'num_joints_in': poses_test_2d[0].shape[-2],
            'in_features': poses_test_2d[0].shape[-1],
            'num_joints_out': poses_test[0].shape[-2],
            'out_features': poses_test[0].shape[-1],
            'hidden_size': args.lstm_hidden_features,
            'num_cells': args.lstm_cells, 
            'head_layers': lstm_head_layers,
            'dropout': args.lstm_dropout
        }

        model_pos_train = CoupledLSTM(**lstm_hyperparams)
        
        model_pos = CoupledLSTM(**lstm_hyperparams)
        
    case 'Transformer':
        receptive_field = 243
        pad = (receptive_field - 1) // 2 # Padding on each side
        if args.causal:
            print('INFO: Using causal convolutions')
            causal_shift = pad
        else:
            causal_shift = 0

        # Parse the transformer head architecture
        transformer_head_layers = [int(x) for x in args.transformer_head_architecture.split(',')]

        transformer_hyperparams = {
            'num_joints_in': poses_test_2d[0].shape[-2],
            'in_features': poses_test_2d[0].shape[-1],
            'num_joints_out': poses_test[0].shape[-2],
            'out_features': poses_test[0].shape[-1],
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'n_heads': args.n_heads,
            'dim_feedforward': args.dim_feedforward,
            'head_layers': transformer_head_layers,
            'dropout': args.transformer_dropout
        }

        model_pos_train = CoupledTransformer(**transformer_hyperparams)

        model_pos = CoupledTransformer(**transformer_hyperparams)
    case 'LSTM-Uncoupled':
        receptive_field = 243
        pad = (receptive_field - 1) // 2 # Padding on each side
        if args.causal:
            print('INFO: Using causal convolutions')
            causal_shift = pad
        else:
            causal_shift = 0

        lstm_head_layers = [int(x) for x in args.lstm_head_architecture.split(',')]
        
        model_pos_train = UncoupledLSTM(num_joints_in = poses_test_2d[0].shape[-2],
                                in_features = poses_test_2d[0].shape[-1],
                                num_joints_out = poses_test[0].shape[-2],
                                out_features = poses_test[0].shape[-1],
                                hidden_size = args.lstm_hidden_features,
                                num_cells = args.lstm_cells, 
                                head_layers = lstm_head_layers,
                                dropout = args.lstm_dropout)
        
        model_pos = UncoupledLSTM(num_joints_in = poses_test_2d[0].shape[-2],
                                in_features = poses_test_2d[0].shape[-1],
                                num_joints_out = poses_test[0].shape[-2],
                                out_features = poses_test[0].shape[-1],
                                hidden_size = args.lstm_hidden_features,
                                num_cells = args.lstm_cells, 
                                head_layers = lstm_head_layers,
                                dropout = args.lstm_dropout)
    case _:
        raise KeyError('Invalid model name')

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    print("INFO: using CUDA")
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

    if args.model_name == "StackedPoselifter":
        model_fcn = model_fcn.cuda()
        model_transformer = model_transformer.cuda()
else:
    print("INFO: CUDA unavailable")
    
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage, weights_only=False)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

test_generator = UnchunkedGenerator(cameras_test, poses_test, poses_test_2d,
                                    pad=pad, causal_shift=causal_shift,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

def train(n_epochs, train_generator, test_generator,
          model_pos_train, model_pos, optimizer,
          save_state=True, plot_losses=True, lr=0.001, lr_decay=0.95):
    
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    if args.resume:
        assert not args.tune_hyperparameters, 'Cannot resume training with hyperparameter tuning'
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        
        lr = checkpoint['lr']
            
    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')

    best_validation_p1_error = float('inf')
    while epoch < n_epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        model_pos_train.train()

        for batch_cam, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_cam = torch.from_numpy(batch_cam.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                inputs_cam = inputs_cam.cuda()
            # inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            if isinstance(model_pos_train, (CamLSTMBase, CamTransformerBase)):
                predicted_3d_pos = model_pos_train(inputs_2d, inputs_cam)
            elif isinstance(model_pos_train, TemporalModelBase):
                predicted_3d_pos = model_pos_train(inputs_2d)
            elif isinstance(model_pos_train, StackedPoseLifter):
                with torch.no_grad():
                    transformer_predicted_3d_pos = model_transformer(inputs_2d, inputs_cam)
                    fcn_predicted_3d_pos = model_fcn(inputs_2d)
                predicted_3d_pos = model_pos_train(transformer_predicted_3d_pos, fcn_predicted_3d_pos)

            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()

            epoch_loss_3d_valid = 0
            N = 0
            
            if not args.no_eval:
                # Evaluate on test set
                for batch_cam, batch_3d, batch_2d, _ in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_cam = torch.from_numpy(batch_cam.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                        inputs_cam = inputs_cam.cuda()
                    # inputs_3d[:, :, 0] = 0

                    # Predict 3D poses
                    if isinstance(model_pos, (CamLSTMBase, CamTransformerBase)):
                        predicted_3d_pos = model_pos.sliding_window(inputs_2d, inputs_cam,
                                                                      test_generator.seq_length)
                    elif isinstance(model_pos, TemporalModelBase):
                        predicted_3d_pos = model_pos(inputs_2d)
                    elif isinstance(model_pos, StackedPoseLifter):
                        transformer_predicted_3d_pos = model_transformer.sliding_window(inputs_2d, inputs_cam,
                                                                        test_generator.seq_length)
                        fcn_predicted_3d_pos = model_fcn(inputs_2d)
                        predicted_3d_pos = model_pos(transformer_predicted_3d_pos.squeeze(),
                                                     fcn_predicted_3d_pos.squeeze())
                        predicted_3d_pos = predicted_3d_pos.view(inputs_3d.shape)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)
                best_validation_p1_error = min(best_validation_p1_error, epoch_loss_3d_valid / N)

        elapsed = (time() - start_time)/60
        
        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_valid[-1] * 1000))
        
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1
        
        # Decay BatchNorm momentum
        if isinstance(model_pos_train, TemporalModelBase):
            momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
            model_pos_train.set_bn_momentum(momentum)
            
        # Save checkpoint if necessary
        if save_state and epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
            }, chk_path)
            
        # Save training curves after every epoch, as .png images (if requested)
        if plot_losses and args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
            
            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d valid'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')
    
    return best_validation_p1_error

if args.tune_hyperparameters:
    config_path = os.path.join(os.path.dirname(__file__), 'gridsearch.json')
    with open(config_path, 'r') as f:
        search_space = json.load(f)

    if args.use_model not in search_space:
        raise ValueError(f"Model '{args.use_model}' not found in gridsearch.json")
    
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)
    
    validation_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
            pad=pad, causal_shift=causal_shift,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    print('INFO: Validation on {} frames'.format(validation_generator.num_frames()))
    
    print(f"Running grid search for model: {args.use_model}")
    best_validation_p1_error = float('inf')
    best_params = None
    
    for idx, params in enumerate(search_space[args.use_model]):
        print(f"  Trial {idx + 1}: {params}")
        hyperparams = {param: value for param, value in params.items() if param not in [
            'learning_rate', 'learning_rate_decay', 'batch_size']}
        
        lr = params['learning_rate'] if 'learning_rate' in params else args.learning_rate
        lr_decay = params['learning_rate_decay'] if 'learning_rate_decay' in params else args.lr_decay
        batch_size = params['batch_size'] if 'batch_size' in params else args.batch_size

        train_generator = ChunkedGenerator(batch_size//args.stride, cameras_train, poses_train,
            poses_train_2d, args.stride, pad=pad, causal_shift=causal_shift, shuffle=True,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

        match args.use_model:
            case 'LSTM_Coupled':

                lstm_hyperparams.update(hyperparams)

                model_pos_train = UncoupledLSTM(**lstm_hyperparams)
                model_pos = UncoupledLSTM(**lstm_hyperparams)
                
            case 'Transformer':

                transformer_hyperparams.update(hyperparams)
                model_pos_train = CoupledTransformer(**transformer_hyperparams)
                model_pos = CoupledTransformer(**transformer_hyperparams)
        
        optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)
            
        best_validation_p1_error = train(args.tuning_epochs, train_generator,
            validation_generator, model_pos_train,
            model_pos, optimizer,
            save_state=False, plot_losses=False, lr=lr, lr_decay=lr_decay)
        
        if best_validation_p1_error < best_validation_p1_error:
            best_validation_p1_error = best_validation_p1_error
            best_params = params

    print('Best validation P1 error:', best_validation_p1_error)
    print('Best hyperparameters:', best_params)
        
elif not args.evaluate:
    print("Training Subjects: ", ", ".join(subjects_train))
    print("Test Subjects: ", ", ".join(subjects_test))
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate

    print(model_pos_train)

    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)
        
    lr_decay = args.lr_decay
    
    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train,
                                       poses_train_2d, args.stride, pad=pad, causal_shift=causal_shift, shuffle=True,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    
    train(args.epochs, train_generator, test_generator, model_pos_train, model_pos, optimizer, save_state=True, plot_losses=True)


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0

    cam_info_per_seq = []
    pose_motion_per_seq = []
    
    # Ensemble metrics
    fcn_diffs = []
    transformer_diffs = []

    e1_per_seq = []

    with torch.no_grad():
        model_pos.eval()

        N = 0

        for cams, batch, batch_2d, seq_info in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_cam = torch.from_numpy(cams.astype('float32'))

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()
                inputs_cam = inputs_cam.cuda()

            # inputs_3d[:, :, 0] = 0

            # Positional model
            if isinstance(model_pos, TemporalModelBase):
                predicted_3d_pos = model_pos(inputs_2d)
            elif isinstance(model_pos, (CamLSTMBase, CamTransformerBase)):
                predicted_3d_pos = model_pos.sliding_window(inputs_2d, inputs_cam, test_generator.seq_length)
            elif isinstance(model_pos, StackedPoseLifter):
                transformer_predicted_3d_pos = model_transformer.sliding_window(inputs_2d, inputs_cam,
                                                                test_generator.seq_length)
                fcn_predicted_3d_pos = model_fcn(inputs_2d)
                predicted_3d_pos = model_pos(transformer_predicted_3d_pos.squeeze(),
                                                fcn_predicted_3d_pos.squeeze())
                predicted_3d_pos = predicted_3d_pos.view(inputs_3d.shape)

                fcn_pred_batch = fcn_predicted_3d_pos.reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1]).cpu().numpy()
                transformer_pred_batch = transformer_predicted_3d_pos.reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1]).cpu().numpy()
                stacked_pred_batch = predicted_3d_pos.reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1]).cpu().numpy()

                fcn_diff = p_mpjpe(fcn_pred_batch, stacked_pred_batch)
                transformer_diff = p_mpjpe(transformer_pred_batch, stacked_pred_batch)
                fcn_diffs.append(fcn_diff.item())
                transformer_diffs.append(transformer_diff.item())
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
            
            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            
            e1_per_seq.append(error.cpu().numpy())
            cam_info_per_seq.append(seq_info)
            pose_motion = np.linalg.norm(np.diff(batch, axis=1), axis=-1)

            pose_motion = np.mean(pose_motion.squeeze(), axis=(0, 1))
            pose_motion_per_seq.append(pose_motion)

            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            p_mpje_loss = p_mpjpe(predicted_3d_pos, inputs)
            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpje_loss

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
    
    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000

    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    # Return metrics and cam v, omega to use for action wise correlation
    return e1, e2, e3, ev, e1_per_seq, cam_info_per_seq, pose_motion_per_seq, fcn_diffs, transformer_diffs


if args.render:
    print('Rendering...')
    print(keypoints[args.viz_subject].keys())
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
        cam_action = dataset[args.viz_subject][args.viz_action]['cameras']
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator([cam_action], [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    
    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)
    
    if args.viz_output is not None:
        
        # Invert camera transformation
        cams = dataset.cameras()[args.viz_subject][args.viz_action]
        cam_intrinsics = cams['intrinsics']
        if ground_truth is not None:
            prediction_frames = list(prediction)
            gt_frames = list(ground_truth)
            world_prediction_frames = []
            world_gt_frames = []
            for i in range(len(prediction_frames)):#
                prediction_frame = prediction_frames[i]
                gt_frame = gt_frames[i]

                if args.dataset == '3DPW':
                    cam_extrinsic = cams['extrinsics'][i]
                    cam_orientation = cam_extrinsic[:, :3]
                    cam_translation = -cam_orientation.T @ cam_extrinsic[:, 3]
                    world_prediction = (prediction_frames[i] @ cam_orientation) + cam_translation
                    world_gt = (gt_frames[i] @ cam_orientation) + cam_translation
                    world_gt = world_gt[:, [0, 2, 1]]  # 3DPW has Y and Z swapped
                    world_prediction = world_prediction[:, [0, 2, 1]]  # 3DPW has Y and Z swapped

                else:
                    prediction_frame[:, 1] = -prediction_frame[:, 1]
                    gt_frame[:, 1] = -gt_frame[:, 1]
                    cam_extrinsic = cams['extrinsics'][i]
                    cam_orientation = cam_extrinsic[:, :3]
                    cam_translation = -cam_orientation.T @ cam_extrinsic[:, 3]
                    world_prediction = (prediction_frames[i] @ cam_orientation) + cam_translation
                    world_gt = (gt_frames[i] @ cam_orientation) + cam_translation

                world_prediction_frames.append(world_prediction)
                world_gt_frames.append(world_gt)

            prediction = np.array(world_prediction_frames)
            ground_truth = np.array(world_gt_frames)
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            ground_truth[:, :, 2] -= np.min(ground_truth[:, :, 2])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        
        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth
        
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=1280, h=720)
        
        from common.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton_2d(), dataset.skeleton_3d(), dataset.fps(), args.viz_bitrate, cam_intrinsics['azimuth'],
                         args.viz_output, limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(int(cam_intrinsics['res_w']), int(cam_intrinsics['res_h'])),
                         input_video_skip=args.viz_skip)
    
else:
    print('Evaluating...')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []
        out_cams = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):
                out_poses_3d.append(poses_3d[i])

            out_cams.append(dataset._cameras[subject][action])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        
        return out_poses_3d, out_poses_2d, out_cams

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        
        fcn_diffs = []
        transformer_diffs = []

        e1_actions = []
        cam_info_actions = []
        pose_motion = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act, cams_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(cams_act, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            e1, e2, e3, ev, e1_per_action, cam_info_per_action, pose_motion_per_action, fcn_diffs, transformer_diffs = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
            
            fcn_diffs += fcn_diffs
            transformer_diffs += transformer_diffs

            e1_actions += e1_per_action
            cam_info_actions += cam_info_per_action
            pose_motion += pose_motion_per_action

        cam_velocities = np.linalg.norm(np.array([cam_info['cam_velocity']
            for cam_info in cam_info_actions]), axis=1)
        cam_accelerations = np.linalg.norm(np.array([cam_info['cam_acceleration'] 
            for cam_info in cam_info_actions]), axis=1)
        cam_angular_velocities = np.linalg.norm(np.array([cam_info['cam_angular_velocity']
            for cam_info in cam_info_actions]), axis=1)
        cam_angular_accelerations = np.linalg.norm(np.array([cam_info['cam_angular_acceleration']
            for cam_info in cam_info_actions]), axis=1)

        pose_motion = np.array(pose_motion)

        e1_actions_arr = np.array(e1_actions)

        # Compute PMCC between e1 and cam movement
        corr_data = np.stack([
            e1_actions_arr,
            cam_velocities,
            cam_accelerations,
            cam_angular_velocities, 
            cam_angular_accelerations,
            pose_motion], axis=1)

        corr_matrix = np.corrcoef(corr_data)
        cam_velocity_r = corr_matrix[0, 1]
        cam_acceleration_r = corr_matrix[0, 2]
        cam_angular_velocity_r = corr_matrix[0, 3]
        cam_angular_acceleration_r = corr_matrix[0, 4]
        pose_motion_r = corr_matrix[0, 5]

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')
        print('PMCC             (MPJPE and cam velocity):', cam_velocity_r)
        print('PMCC         (MPJPE and cam acceleration):', cam_acceleration_r)
        print('PMCC     (MPJPE and cam angular velocity):', cam_angular_velocity_r)
        print('PMCC (MPJPE and cam angular acceleration):', cam_angular_acceleration_r)
        print('PMCC              (MPJPE and Pose Motion):', pose_motion_r)

        if args.model_name == 'StackedPoselifter':
            print('FCN Diff                        (P-MPJPE):', np.mean(fcn_diffs), 'mm')
            print('Transformer Diff                (P-MPJPE):', np.mean(transformer_diffs), 'mm')

    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
