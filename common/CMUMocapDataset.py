import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates


h36m_skeleton_nonstatic = Skeleton(
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    joints_left=[4, 5, 6, 11, 12, 13],
    joints_right=[1, 2, 3, 14, 15, 16]
)

class CMUMocapDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        """
        Initializes the dataset.
        
        Args:
            positions_path (str): Path to the npz file containing the 'positions_3d' dict.
            extrinsics_path (str): Path to the npz file containing the 'cam_extrinsics' dict.
            remove_static_joints (bool): (Unused here) whether to remove static joints.
        """
        super().__init__(fps=240, skeleton=h36m_skeleton_nonstatic)
        data = np.load(path, allow_pickle=True)

        pose_data = data['positions_3d'].item()
        cam_extrinsics = data['cam_extrinsics'].item()

        self._data = {}
        self._cameras = {}

        CMU_cam_intrinsic = {
            'id': '1',
            'center': np.array([0, 0], dtype='float32'),
            'focal_length': np.array([1000, 1000], dtype='float32'),
            'radial_distortion': np.array([0, 0, 0], dtype='float32'),
            'tangential_distortion': np.array([0, 0], dtype='float32'),
            'res_w': 1280,
            'res_h': 720,
            'azimuth': 0,  # Only used for visualization
        }

        # Normalise camera frame
        norm_center = normalize_screen_coordinates(CMU_cam_intrinsic['center'],
                                                    CMU_cam_intrinsic['res_w'],
                                                    CMU_cam_intrinsic['res_h'])
        CMU_cam_intrinsic['center'] = norm_center.astype('float32')
        CMU_cam_intrinsic['focal_length'] = 2 * CMU_cam_intrinsic['focal_length'] / CMU_cam_intrinsic['res_w']

        # Add intrinsic parameters vector
        self.cam_intrinsics = np.concatenate((
            CMU_cam_intrinsic['focal_length'],
            CMU_cam_intrinsic['center'],
            CMU_cam_intrinsic['radial_distortion'],
            CMU_cam_intrinsic['tangential_distortion']
        ))

        for subject, actions in pose_data.items():
            self._data[subject] = {}
            self._cameras[subject] = {}
            for action_name, positions in actions.items():
                extrinsics_for_action = cam_extrinsics[subject][action_name]
                num_frames = positions.shape[0]
                assert len(extrinsics_for_action) == num_frames, (
                    f"Number of extrinsics ({len(extrinsics_for_action)}) does not match number of frames ({num_frames}) "
                    f"for subject {subject} action {action_name}"
                )
                cameras_for_action = []

                for i in range(num_frames):
                    frame_camera = copy.deepcopy(CMU_cam_intrinsic)
                    # Update the copied intrinsics with the frame-specific extrinsics.
                    frame_camera.update(extrinsics_for_action[i])
                    cameras_for_action.append(frame_camera)
                # Store positions and the per-frame cameras in the same structure.
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': cameras_for_action
                }
                self._cameras[subject][action_name] = cameras_for_action

    def supports_semi_supervised(self):
        return False
