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
from common.datasets.ThreeDPWDataset import ThreeDPWDataset
from common.camera import world_to_camera, image_coordinates, normalize_screen_coordinates, project_to_2d, project_to_2d_linear
from common.utils import wrap

output_filename = 'data_3d_3dpw'
output_filename_2d = 'data_2d_3dpw_detections'

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
        output_2d_poses = {}
        cam_seqs = {}
        cam_intrinsics = {}

        for subject in tqdm(subjects):
            positions_3d[subject] = {}
            output_2d_poses[subject] = {}
            cam_seqs[subject] = {}
            cam_intrinsics[subject] = {}
            # if benchmark flag set, only convert .pkl files containing _benchmark
            file_list = glob(args.from_source + '/' + subject + '/*.pkl')
            
            for f in file_list:
                print(f)
                scene_file = open(f, "rb")
                scene_data = pickle.load(scene_file, encoding='latin')

                joint_positions_subjects = np.array(scene_data['jointPositions'])
                joint_detections_subjects = np.array(scene_data['poses2d'])
                
                num_subjects = joint_positions_subjects.shape[0]
                
                cam_seq = scene_data['cam_poses']
                intrinsic_mat = scene_data['cam_intrinsics']
                for i in range(num_subjects):
                    joint_positions = joint_positions_subjects[i]
                    joint_detections = joint_detections_subjects[i][:, :2, :] # filter confidence scores

                    joint_positions = joint_positions.reshape(-1, 24, 3).astype('float32')
                    joint_detections = joint_detections.reshape(-1, 18, 2).astype('float32')

                    print(joint_positions.shape)

                    positions_hom = np.concatenate([joint_positions, 
                        np.ones((joint_positions.shape[0], joint_positions.shape[1], 1), dtype='float32')], axis=2)

                    cam_positions = np.einsum('tij,tnj->tni', cam_seq, positions_hom)

                    cam_positions[:, :, 1] = -cam_positions[:, :, 1]

                    action_name = f.removesuffix('.pkl') + f'_{i}'

                    positions_3d[subject][action_name] = cam_positions
                    cam_seqs[subject][action_name] = cam_seq
                    cam_intrinsics[subject][action_name] = intrinsic_mat

                    output_2d_poses[subject][action_name] = joint_detections


        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=positions_3d, cam_seqs=cam_seqs, cam_intrinsics=cam_intrinsics)
        dataset = ThreeDPWDataset(output_filename + '.npz')
        metadata = {
            'num_joints': dataset.skeleton_2d().num_joints(),
            'keypoints_symmetry': [dataset.skeleton_2d().joints_left(), dataset.skeleton_2d().joints_right()],
            'layout_name': 'coco'
        }
        np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
        print('Done.')
            
    else:
        print('Please specify the dataset source')
        exit(0)
        
