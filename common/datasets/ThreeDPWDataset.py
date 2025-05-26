import numpy as np
import copy
from common.skeleton import Skeleton
from common.datasets.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
from scipy.linalg import logm

def skew_to_vec(omega_hat):
    return np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])

coco_skeleton = Skeleton(
    parents = [-1, 0, 1, 2, 3, 1, 2, 3, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15],
    joints_left = [2, 3, 4, 8, 9, 10, 14, 16],
    joints_right = [5, 6, 7, 11, 12, 13, 15, 17]
)

smpl_skeleton = Skeleton(
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
             16, 17, 18, 19, 20, 21],
    joints_left = [2, 5, 8, 11, 14, 17, 19, 21, 23],
    joints_right = [1, 4, 7, 10, 13, 16, 18, 20, 22]
)

class ThreeDPWDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        """
        Initializes the dataset.
        
        Args:
            positions_path (str): Path to the npz file containing the 'positions_3d' dict.
            extrinsics_path (str): Path to the npz file containing the 'cam_extrinsics' dict.
            remove_static_joints (bool): (Unused here) whether to remove static joints.
        """
        super().__init__(fps=60, skeleton_2d=coco_skeleton, skeleton_3d=smpl_skeleton)
        data = np.load(path, allow_pickle=True)

        pose_data = data['positions_3d'].item()
        cam_seqs = data['cam_seqs'].item()
        cam_intrinsics = data['cam_intrinsics'].item()

        self._data = {}
        self._cameras = {}

        for subject, actions in pose_data.items():
            self._data[subject] = {}
            self._cameras[subject] = {}
            for action_name, positions in actions.items():
                cam_seq = cam_seqs[subject][action_name]
                intrinsic_mat = cam_intrinsics[subject][action_name]

                num_frames = positions.shape[0]
                assert len(cam_seq) == num_frames, (
                    f"Number of extrinsics ({len(cam_seq)}) does not match number of frames ({num_frames}) "
                    f"for subject {subject} action {action_name}"
                )

                cam_translations = []
                cam_rotations = []

                for cam_ext in cam_seq:
                    R = cam_ext[:3, :3].T
                    t = -R @ cam_ext[:3, 3]

                    cam_translations.append(t)
                    cam_rotations.append(R)

                cam_velocities = np.diff(np.array(cam_translations), axis=0) * self.fps()
                cam_accelerations = np.diff(cam_velocities, axis=0) * self.fps()

                angular_velocities = []
                for i in range(len(cam_rotations) - 1):
                    R_i = cam_rotations[i]
                    R_next = cam_rotations[i+1]
                    delta_R = R_i.T @ R_next
                    log_delta_R = logm(delta_R)
                    omega_hat = log_delta_R * self.fps()
                    omega = skew_to_vec(omega_hat)
                    angular_velocities.append(omega)
                
                angular_accelerations = np.diff(np.array(angular_velocities), axis=0) * self.fps()

                avg_cam_velocity = np.mean(cam_velocities, axis=0)
                avg_cam_acceleration = np.mean(cam_accelerations, axis=0)
                avg_cam_angular_velocity = np.mean(angular_velocities, axis=0)
                avg_cam_angular_acceleration = np.mean(angular_accelerations, axis=0)

                cameras_for_action = {'intrinsics': intrinsic_mat,
                                      'extrinsics': cam_seq,
                                      'cam_velocity': avg_cam_velocity,
                                      'cam_acceleration': avg_cam_acceleration,
                                      'cam_angular_velocity': avg_cam_angular_velocity,
                                      'cam_angular_acceleration': avg_cam_angular_acceleration}

                # Store positions and the per-frame cameras in the same structure.
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': cameras_for_action
                }
                self._cameras[subject][action_name] = cameras_for_action

    def supports_semi_supervised(self):
        return False
