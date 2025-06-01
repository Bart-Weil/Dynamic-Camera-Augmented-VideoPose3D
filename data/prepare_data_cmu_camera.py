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
from common.datasets.CMUMocapDataset import CMUMocapDataset
from common.camera import world_to_camera, image_coordinates, normalize_screen_coordinates, project_to_2d, project_to_2d_linear
from common.utils import wrap

output_filename = '/vol/bitbucket/bw1222/data/npz/data_3d_CMU'
output_filename_2d = '/vol/bitbucket/bw1222/data/npz/data_2d_CMU_gt'

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
        output_2d_poses = {}

        for subject in tqdm(subjects):
            positions_3d[subject] = {}
            cam_seqs[subject] = {}
            output_2d_poses[subject] = {}
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
                positions_2d = scene_data['pose_2d'][:, :, :2].reshape(-1, 17, 2).astype('float32')

                # positions_hom = np.concatenate([positions, 
                #    np.ones((positions.shape[0], positions.shape[1], 1), dtype='float32')], axis=2)

                # cam_positions = np.einsum('tij,tnj->tni', scene_data['cam_sequence']['cam_extrinsic'], positions_hom)

                # cam_positions[:, :, 1] = -cam_positions[:, :, 1]

                positions_3d[subject][f] = positions #cam_positions
                output_2d_poses[subject][f] = positions_2d
                cam_seqs[subject][f] = scene_data['cam_sequence']
                
                   

        print('Saving...')
        np.savez_compressed(output_filename, positions_3d=positions_3d, cam_seqs=cam_seqs)

        dataset = CMUMocapDataset(output_filename + '.npz')
        metadata = {
            'num_joints': dataset.skeleton_2d().num_joints(),
            'keypoints_symmetry': [dataset.skeleton_2d().joints_left(), dataset.skeleton_2d().joints_right()],
            'layout_name': 'h36m'
        }

        np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
        print('Done.')
            
    else:
        print('Please specify the dataset source')
        exit(0)

