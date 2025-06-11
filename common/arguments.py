# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('--subjects-test', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=10, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')

    # Model selection
    parser.add_argument('--use-model', dest='model_name', default='FCN', type=str, help='FCN, LSTM-Uncoupled, Transformer, LSTM-Coupled')
    # Learning arguments
    parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
    
    # LSTM arguments
    parser.add_argument('--hidden-features', dest='lstm_hidden_features', default=128, type=int, metavar='N', help='number of hidden features for LSTM')
    parser.add_argument('--lstm-cells', dest='lstm_cells', default=2, type=int, metavar='N', help='number of stacked LSTM cells')
    parser.add_argument('--lstm-head-architecture', dest='lstm_head_architecture', default='100', type=str, metavar='X,Y,Z', 
                        help='layer sizes for LSTM head separated by comma')
    parser.add_argument('--lstm-dropout', dest='lstm_dropout', default=0.25, type=float, metavar='P', help='LSTM dropout probability')

    # Transformer arguments
    parser.add_argument('--d-model', dest='d_model', default=128, type=int, metavar='N', help='transformer embedding dimension')
    parser.add_argument('--num-layers', dest='num_layers', default=2, type=int, metavar='N', help='number of transformer encoder layers')
    parser.add_argument('--n_heads', dest='n_heads', default=4, type=int, metavar='N', help='number of attention heads')
    parser.add_argument('--dim-feedforward', dest='dim_feedforward', default=128, type=int, metavar='N', help='feedforward network dimension in transformer layers')
    parser.add_argument('--transformer-head-architecture', dest='transformer_head_architecture', default='128,128,128', type=str, metavar='X,Y,Z', 
                        help='layer sizes for transformer head separated by comma')
    parser.add_argument('--transformer-dropout', dest='transformer_dropout', default=0.25, type=float, metavar='P', help='transformer dropout probability')

    # Temporal FCN arguments
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('--fcn-architecture', dest='fcn_architecture', default='3,3,3,3,3', type=str, metavar='LAYERS', help='temporal FCN filter widths separated by comma')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')
    parser.add_argument('--fcn-dropout', dest='fcn_dropout', default=0.25, type=float, metavar='P', help='temporal FCN dropout probability')

    # Stacked Pose Lifter arguments
    parser.add_argument('--stacked-num-layers', dest='stacked_num_layers', default=3, type=int, metavar='N', help='number of stacked layers in StackedPoseLifter')
    parser.add_argument('--layer-size', dest='layer_size', default=256, type=int, metavar='N', help='number of features in each layer of StackedPoseLifter')
    parser.add_argument('--stacked-pose-lifter-dropout', dest='stacked_pose_lifter_dropout', default=0.25, type=float, metavar='P', help='dropout probability in StackedPoseLifter')
    parser.add_argument('--transformer-weights', dest='transformer_weights', default='', type=str, metavar='PATH', help='path to pre-trained transformer weights for StackedPoseLifter')
    parser.add_argument('--fcn-weights', dest='fcn_weights', default='', type=str, metavar='PATH', help='path to pre-trained FCN weights for StackedPoseLifter')

    # Experimental
    parser.add_argument('--tune-hyperparameters', action='store_true', help='enable hyperparmeter tuning (search spaces per model defined at root/gridsearch.json)')
    parser.add_argument('-te', '--tuning-epochs', default=20, type=int, metavar='N', help='number of epochs for hyperparameter tuning')
    parser.add_argument('--subjects-validate', type=str, metavar='LIST', help='validation subjects separated by comma')
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int, metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-optimizations', action='store_true', help='disable optimized model for single-frame predictions')
    
    # Visualization
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')
    
    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()
        
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args
