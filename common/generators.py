# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np

class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    intrinsics -- list of camera intrinsic (world-to-camera) matrices (per sequence)
    extrinsics -- array of camera extrinsic (world-to-camera) matrices
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cams, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert len(extrinsics) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:])

        self.seq_length = chunk_length + 2*pad

        # Initialize buffers
        self.batch_cam = np.empty((batch_size, self.seq_length, 3, 4))
        self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, self.seq_length, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cams = cams
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
        
    def pad_chunk(self, kps, start, end):
        min_idx = max(start, 0)
        max_idx = min(end, kps.shape[0])
        pad_left = min_idx - start
        pad_right = end - max_idx
        if pad_left != 0 or pad_right != 0:
            return np.pad(kps[min_idx:max_idx], ((pad_left, pad_right), (0, 0), (0, 0)), 'edge')
        else:
            return kps[min_idx:max_idx]

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift
                    self.batch_2d[i] = self.pad_chunk(self.poses_2d[seq_i], start_2d, end_2d)
                    
                    batch_extrinsics = self.pad_chunk(self.cams[seq_i]['extrinsics'], start_2d, end_2d)

                    cam_dict = self.cams[seq_i]['intrinsics']
                    fx, fy = cam_dict['focal_length']
                    cx, cy = cam_dict['center']

                    seq_intrinsic_mat = np.array([
                        [fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,   1]
                    ], dtype=np.float32)
                    # Compute camera matrices
                    self.batch_cam[i] = seq_intrinsic_mat @ batch_extrinsics

                    self.batch_3d[i] = self.pad_chunk(self.poses_3d[seq_i], start_3d, end_3d)

                if self.endless:
                    self.state = (b_i + 1, pairs)
                
                yield self.batch_cam, self.batch_3d, self.batch_2d
            
            if self.endless:
                self.state = None
            else:
                enabled = False
            

class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cams, poses_3d, poses_2d, pad=0, causal_shift=0,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None):

        assert len(poses_3d) == len(poses_2d)

        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cams = cams
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        self.seq_length = 1 + 2*pad
 
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def next_epoch(self):
        for idx, (cam_seq, seq_3d, seq_2d) in enumerate(zip_longest(self.cams, self.poses_3d, self.poses_2d)):
            cam_dict = cam_seq['intrinsics']
            fx, fy = cam_dict['focal_length']
            cx, cy = cam_dict['center']

            seq_intrinsic_mat = np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,   1]
            ], dtype=np.float32)

            cam_mat_seq = seq_intrinsic_mat @ cam_seq['extrinsics']

            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                        ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                        'edge'), axis=0)
            batch_cam = np.expand_dims(np.pad(cam_mat_seq,
                        ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                        'edge'), axis=0)

            yield batch_cam, batch_3d, batch_2d, {
                'cam_velocity': cam_seq['cam_velocity'],
                'cam_acceleration': cam_seq['cam_acceleration'],
                'cam_angular_velocity': cam_seq['cam_angular_velocity'],
                'cam_angular_acceleration': cam_seq['cam_angular_acceleration']}
