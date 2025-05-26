# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import numpy as np
import pickle

from tqdm import tqdm

from glob import glob
from shutil import rmtree

import sys
sys.path.append('../')
from common.CMUMocapDataset import CMUMocapDataset
from common.camera import world_to_camera, image_coordinates, normalize_screen_coordinates, project_to_2d, project_to_2d_linear
from common.utils import wrap

output_filename = 'data_3d_CMU'
output_filename_2d = 'data_2d_CMU_gt'

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='CMU Camera dataset downloader/converter')
    
    parser.add_argument('--from-source', default='', type=str, metavar='PATH', help='convert original dataset')
    parser.add_argument('--convert-benchmark', action='store_true', help='include benchmark files')
    
    args = parser.parse_args()
   
    if args.from_source:
        subjects = os.listdir(args.from_source)
        print('Converting original CMU Cameras dataset from', args.from_source)
        positions_3d = {}
        cam_seqs = {}

        for subject in tqdm(subjects):
            positions_3d[subject] = {}
            cam_seqs[subject] = {}
            # if benchmark flag set, only convert .pkl files containing _benchmark
            file_list = glob(args.from_source + '/' + subject + '/*.pkl')
            if args.convert_benchmark:
                file_list = filter(lambda x: '_benchmark' in x, file_list)
            else:
                file_list = filter(lambda x: '_benchmark' not in x, file_list)
            
            for f in file_list:
                scene_file = open(f, "rb")
                scene_data = pickle.load(scene_file)
                positions = scene_data['pose_3d'].reshape(-1, 17, 3).astype('float32')

                positions_hom = np.concatenate([positions, 
                    np.ones((positions.shape[0], positions.shape[1], 1), dtype='float32')], axis=2)

                cam_positions = np.einsum('tij,tnj->tni', scene_data['cam_sequence']['cam_extrinsic'], positions_hom)

                cam_positions[:, :, 1] = -cam_positions[:, :, 1]

                positions_3d[subject][f] = cam_positions
                cam_seqs[subject][f] = scene_data['cam_sequence']
                

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=positions_3d, cam_seqs=cam_seqs)
        
        print('Done.')
            
    else:
        print('Please specify the dataset source')
        exit(0)
        
    dataset = CMUMocapDataset(output_filename + '.npz')

    print('')
    print('Computing ground-truth 2D poses...')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            # Open scene file if needed
            with open(action, "rb") as scene_file:
                scene_data = pickle.load(scene_file)
            positions_2d = []
            cam_pose_frames = list(anim['positions'])
            num_frames = len(cam_pose_frames)
            intrinsics = anim['cameras']['intrinsics']
            # Process each frame individually
            for i in range(num_frames):
                pos_3d_frame = np.array([cam_pose_frames[i]]).astype('float32')
                # dataset.cam_intrinsics is a stacked numpy array of intrinsics used for project_to_2d_linear
                pos_2d = wrap(project_to_2d_linear, pos_3d_frame, dataset.cam_intrinsics, unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(np.array([pos_2d]), w=intrinsics['res_w'], h=intrinsics['res_h'])
                pos_2d_pixel_space = np.squeeze(pos_2d_pixel_space)

                pos_2d_pixel_space += np.array([intrinsics['res_w']/2, intrinsics['res_h']/2])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))

            output_2d_poses[subject][action] = np.array(positions_2d).astype('float32').reshape(-1, 17, 2)

    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()],
        'layout_name': 'h36m'
    }
    np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
    
    print('Done.')
