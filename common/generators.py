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
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, intrinsics, extrinsics, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert len(extrinsics) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        self.batch_cam = np.empty((batch_size, chunk_length + 2*pad, 3, 4))
        self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.augment = augment
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
        
    def augment_enabled(self):
        return self.augment
    
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
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift
                    self.batch_2d[i] = self.pad_chunk(self.poses_2d[seq_i], start_2d, end_2d)
                    
                    batch_extrinsics = self.pad_chunk(self.extrinsics[seq_i], start_2d, end_2d)

                    cam_dict = self.intrinsics[seq_i]
                    fx, fy = cam_dict['focal_length']
                    cx, cy = cam_dict['center']

                    seq_intrinsic_mat = np.array([
                        [fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,   1]
                    ], dtype=np.float32)
                    # Compute camera matrices
                    self.batch_cam[i] = seq_intrinsic_mat @ batch_extrinsics

                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

                    self.batch_3d[i] = self.pad_chunk(self.poses_3d[seq_i], start_3d, end_3d)

                    if flip:
                        # Flip 3D joints
                        self.batch_3d[i, :, :, 0] *= -1
                        self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                self.batch_3d[i, :, self.joints_right + self.joints_left]

                        # Flip horizontal distortion coefficients
                        self.batch_cam[i, 2] *= -1
                        self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                
                yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False
            

class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, intrinsics, extrinsics, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):

        assert len(poses_3d) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_extrinsics, seq_3d, seq_2d in zip_longest(self.extrinsics, self.poses_3d, self.poses_2d):
            batch_cam = self.intrinsics @ seq_extrinsics

            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                        ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                        'edge'), axis=0)
        
            # Ignore for now, requires expanded seq_cam dimensions (since removed)
            # if self.augment:
            #     # Append flipped version
            #     if batch_cam is not None:
            #         batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
            #         batch_cam[1, 2] *= -1
            #         batch_cam[1, 7] *= -1
                
            #     if batch_3d is not None:
            #         batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
            #         batch_3d[1, :, :, 0] *= -1
            #         batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

            #     batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
            #     batch_2d[1, :, :, 0] *= -1
            #     batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d