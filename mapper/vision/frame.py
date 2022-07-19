import numpy as np

import mapper.vision.flow as flow
import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.tracking as trk
import mapper.vision.transform as trf


class Frame:
    def __init__(self, frame_id: int, image: np.ndarray,
                 intrinsic_mat: np.ndarray, parent_keyframe):
        assert frame_id >= 0
        assert im.is_image(image)
        assert im.num_channels(image) == 1
        assert isinstance(intrinsic_mat, np.ndarray)
        assert intrinsic_mat.shape == (3, 3)

        print(f'Create frame={frame_id}')

        self.frame_id = frame_id
        self.image = image
        self.intrinsic_mat = intrinsic_mat
        self.pose_mat = mat.pose_matrix(
            np.eye(3, 3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        self.parent_keyframe = parent_keyframe

        # Stuff only properly populated for keyframes.
        self.is_keyframe = False
        self.train_points = None
        self.depth_map = None
        self.heat_map = None

    def track_and_map(self) -> None:
        assert not self.is_keyframe

        if not self.parent_keyframe is None:
            assert self.parent_keyframe.is_keyframe

            # Track against the parent keyframe.
            self.__track()

    def should_be_promoted(self) -> bool:
        if not self.parent_keyframe is None:
            assert self.parent_keyframe.is_keyframe

            ypr_key, t_key = mat.decomp_pose_matrix_yprt_yxz(
                self.parent_keyframe.pose_mat)
            ypr, t = mat.decomp_pose_matrix_yprt_yxz(self.pose_mat)

            ypr_diff = np.sum(np.abs(np.array(ypr_key) - np.array(ypr)))
            t_diff = np.linalg.norm(t_key - t)

            return ypr_diff > 4 or t_diff > 3
        else:
            return True

    def promote_to_keyframe(self) -> None:
        print(f'Frame={self.frame_id} is promoted to keyframe')
        self.is_keyframe = True

        self.train_points = flow.points_to_track(self.image)
        print(f' {len(self.train_points)} image points are captured')

        self.depth_map = np.ones_like(self.image, dtype=np.float64)
        self.heat_map = np.zeros_like(self.image, dtype=np.int8)

    def __track(self) -> tuple:
        keyframe = self.parent_keyframe
        print(
            f'Track frame={self.frame_id} to keyframe={keyframe.frame_id}')

        # Compute optical flow.
        train_match, query_match = flow.sparse_optical_flow(keyframe.image,
                                                            self.image,
                                                            keyframe.train_points)
        print(f' {len(train_match)} matching points in optical flow')

        # Compute pose estimation.
        rel_pose, pose_train_match, query_train_match, _ = trk.visual_pose_prediction_plk(
            train_match, query_match, self.intrinsic_mat)
        print(f' {len(pose_train_match)} matching points in pose estimation')

        # Change pose.
        self.pose_mat = trf.change_pose(self.pose_mat, rel_pose)

        return pose_train_match, query_train_match
