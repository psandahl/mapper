import numpy as np

import mapper.vision.flow as flow
import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.tracking as trk
import mapper.vision.transform as trf


class Frame:
    def __init__(self, frame_id: int, image: np.ndarray,
                 intrinsic_mat: np.ndarray):
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

        self.train_points = flow.points_to_track(self.image)

        self.is_keyframe = False

        # Stuff only properly populated for keyframes.
        self.depth_map = None
        self.heat_map = None

    def track_against(self, other) -> None:
        assert not other is None
        assert not self.is_keyframe

        print(
            f'Track frame={self.frame_id} against frame={other.frame_id}')

        train_match, query_match = flow.sparse_optical_flow(other.image,
                                                            self.image,
                                                            other.train_points)
        print(f' {len(train_match)} matching points in optical flow')

        rel_pose, pose_train_match, _, _ = trk.visual_pose_prediction_plk(train_match,
                                                                          query_match,
                                                                          self.intrinsic_mat)
        print(f' {len(pose_train_match)} matching points in pose estimation')

        self.pose_mat = trf.change_pose(other.pose_mat, rel_pose)

    def map_using(self, other) -> None:
        assert self.is_keyframe
        assert not other is None

    def should_be_promoted(self, keyframe) -> bool:
        if not keyframe is None:
            assert keyframe.is_keyframe

            ypr_key, t_key = mat.decomp_pose_matrix_yprt_yxz(
                keyframe.pose_mat)
            ypr, t = mat.decomp_pose_matrix_yprt_yxz(self.pose_mat)

            ypr_diff = np.sum(np.abs(np.array(ypr_key) - np.array(ypr)))
            t_diff = np.linalg.norm(t_key - t)

            return ypr_diff > 10 or t_diff > 8

        else:
            return True

    def promote_to_keyframe(self, keyframe) -> None:
        print(f'Frame={self.frame_id} is promoted to keyframe')
        self.is_keyframe = True

        self.depth_map = np.ones_like(self.image, dtype=np.float64)
        self.heat_map = np.zeros_like(self.image, dtype=np.int8)
