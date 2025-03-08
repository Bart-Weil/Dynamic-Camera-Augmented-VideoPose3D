# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import zipfile
import numpy as np
import pickle
import copy

from tqdm import tqdm

import transforms3d

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
    
    args = parser.parse_args()
   
    if args.from_source:
        subjects = os.listdir(args.from_source)
        print('Converting original CMU Cameras dataset from', args.from_source)
        positions_3d = {}
        cam_extrinsics = {}

        for subject in tqdm(subjects[:1]):
            positions_3d[subject] = {}
            cam_extrinsics[subject] = {}
            file_list = glob(args.from_source + '/' + subject + '/*.pkl')
            
            for f in file_list:
                scene_file = open(f, "rb")
                scene_data = pickle.load(scene_file)
                position_frames = list(scene_data['pose_3d'].reshape(-1, 17, 3).astype('float32'))

                num_frames = len(position_frames)

                orientations = list(scene_data['cam_sequence']['cam_extrinsic'].reshape(-1, 3, 4)[:, :, :3])
                opt_centers = list(scene_data['cam_sequence']['opt_center'].reshape(-1, 1, 3).astype('float32'))

                extrinsic_dicts = [{'orientation': orientations[i],
                                    'translation': opt_centers[i]} for i in range(num_frames)]
                
                cam_position_frames = []
                for i in range(num_frames):
                    # VideoPose3D camera functions expect an array of frames
                    thing = list(orientations[i])
                    # print(transforms3d.euler.quat2euler(thing))
                    cam_position_frame = (position_frames[i] - opt_centers[i]) @ orientations[i].T
                    cam_position_frame[:, 1] = -cam_position_frame[:, 1]
                    # cam_position_frame = wrap(world_to_camera, position_frames[i], orientations[i], opt_centers[i], unsqueeze=True)
                    # print(position_frames[i][0, :])
                    cam_position_frames.append(cam_position_frame.astype('float32'))

                positions_3d[subject][f] = np.array(cam_position_frames)
                cam_extrinsics[subject][f] = extrinsic_dicts

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=positions_3d, cam_extrinsics=cam_extrinsics)
        
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
            # Process each frame individually
            for i in range(num_frames):
                pos_3d_frame = np.array([cam_pose_frames[i]]).astype('float32')
                cam_dict = anim['cameras'][i]
                intrinsics = dataset.cam_intrinsics
                pos_2d = wrap(project_to_2d_linear, pos_3d_frame, intrinsics, unsqueeze=True)
                # intrinsic_mat = np.array([
                #     [intrinsics[0], 0,             intrinsics[2]],
                #     [0,             intrinsics[1], intrinsics[3]],
                #     [0,             0,             1]])
                # projected_frame = pos_3d_frame @ intrinsic_mat.T
                # pos_2d = projected_frame[:, :2] / projected_frame[:, 2, np.newaxis]
                # print(pos_2d.shape)
                pos_2d_pixel_space = image_coordinates(np.array([pos_2d]), w=cam_dict['res_w'], h=cam_dict['res_h'])
                pos_2d_pixel_space = np.squeeze(pos_2d_pixel_space)
                # pos_2d_pixel_space[:, 1] = -pos_2d_pixel_space[:, 1]
                pos_2d_pixel_space += np.array([cam_dict['res_w']/2, cam_dict['res_h']/2])
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
