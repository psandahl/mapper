import cv2 as cv
import numpy as np

from mapper.vision.depthmap import DepthMap
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

        self.train_points = flow.points_to_track(self.image, distance=8)

        self.is_keyframe = False

        # Depth map is only populated for keyframes.
        self.depth_map = None

        print('')

    def track_against(self, other) -> None:
        """
        Track this frame against the other frame.
        """
        assert not other is None
        assert not self.is_keyframe

        print(
            f'Track frame={self.frame_id} against frame={other.frame_id}')

        # Get matching points from optical flow.
        train_match, query_match = flow.sparse_optical_flow(other.image,
                                                            self.image,
                                                            other.train_points)
        print(f' {len(train_match)} matching points from optical flow')

        # Compute relative pose estimation from matching points.
        rel_pose, pose_train_match, _, _ = trk.visual_pose_prediction_plk(train_match,
                                                                          query_match,
                                                                          self.intrinsic_mat)
        print(f' {len(pose_train_match)} matching points in pose estimation')

        # Apply relative pose to the current pose.
        self.pose_mat = trf.change_pose(other.pose_mat, rel_pose)

        print('')

    def map_with(self, other) -> None:
        """
        Use the other frame to do mapping with this keyframe.
        """
        assert self.is_keyframe
        assert not other is None
        assert not other.is_keyframe

        print(f'Map frame={self.frame_id} with frame={other.frame_id}')

        # Create projection matrix for self (i.e. the keyframe).
        R_key, t_key = mat.decomp_pose_matrix(self.pose_mat)
        extrinsic_key = mat.extrinsic_matrix(R_key, t_key)
        projection_key = mat.projection_matrix(
            self.intrinsic_mat, extrinsic_key)

        # Create projection matrix for other frame.
        R_oth, t_oth = mat.decomp_pose_matrix(other.pose_mat)
        extrinsic_oth = mat.extrinsic_matrix(R_oth, t_oth)
        projection_oth = mat.projection_matrix(
            other.intrinsic_mat, extrinsic_oth)

        # Get matching points from optical flow.
        train_match, query_match = flow.sparse_optical_flow(
            self.image, other.image, self.train_points)

        print(f' {len(train_match)} matching points from optical flow')

        # Triangulate the points.
        points = cv.triangulatePoints(projection_key, projection_oth,
                                      train_match.T, query_match.T)
        points = np.transpose(points[:3] / points[3])

        # Iterate the points and use the inliers for depth.
        inliers = 0
        for index, point in enumerate(points):
            # Inlier check 1, points must be in front of cameras.
            if trf.infront_of_camera(extrinsic_key, point) and trf.infront_of_camera(extrinsic_oth, point):
                px_key, z_key = trf.project_point(projection_key, point)
                px_oth, _ = trf.project_point(projection_oth, point)

                # Inlier check 2. Reprojection error must be small.
                err_key = np.sum(np.square(px_key - train_match[index]))
                err_oth = np.sum(np.square(px_oth - query_match[index]))
                if err_key < 0.5 and err_oth < 0.5:
                    # Image coordinate for this sample - use the keypoint instead
                    # of the reprojected point.
                    u, v = train_match[index]

                    # Calculate the inverse depth for this sample.
                    inv_depth = 1.0 / z_key

                    # Add the sample to the depth map.
                    self.depth_map.add((u, v), inv_depth)

                    inliers += 1

        print(f' {inliers} inliers after mapping')
        print('')

    def should_be_promoted(self, keyframe) -> bool:
        """
        Check this frame against the given keyframe, and decide
        if this frame should be promoted to next keyframe
        """
        assert not self.is_keyframe

        if not keyframe is None:
            assert keyframe.is_keyframe

            ypr_key, t_key = mat.decomp_pose_matrix_yprt_yxz(
                keyframe.pose_mat)
            ypr, t = mat.decomp_pose_matrix_yprt_yxz(self.pose_mat)

            ypr_diff = np.sum(np.abs(np.array(ypr_key) - np.array(ypr)))
            t_diff = np.linalg.norm(t_key - t)

            angular_diff = 10.0
            translate_diff = 8.0

            return ypr_diff > angular_diff or t_diff > translate_diff
        else:
            return True

    def promote_to_keyframe(self, keyframe) -> None:
        """
        Promote this frame to keyframe, and inherit depth information
        from the given keyframe.
        """
        assert not self.is_keyframe

        print(f'Frame={self.frame_id} is promoted to keyframe')
        self.is_keyframe = True

        if not keyframe is None:
            assert keyframe.is_keyframe
            assert im.image_size(self.image) == im.image_size(keyframe.image)

            print(f' Inherit depth information from frame={keyframe.frame_id}')

            # Create inverse matrices for the keyframe (note: pose is equal
            # to inversed extrinsic matrix).
            intrinsic_key = np.linalg.inv(keyframe.intrinsic_mat)
            extrinsic_key = keyframe.pose_mat

            # Create extrinsic matrix for self.
            R, t = mat.decomp_pose_matrix(self.pose_mat)
            extrinsic_self = mat.extrinsic_matrix(R, t)

            # Get the exported depth map.
            self.depth_map = keyframe.depth_map.export_to_projection(
                intrinsic_key,
                extrinsic_key,
                self.intrinsic_mat,
                extrinsic_self,
                im.image_size(keyframe.image)
            )

            print(
                f'  {len(keyframe.depth_map.map)} points provided, {len(self.depth_map.map)} points inherited')
        else:
            # No inherit. Create empty map.
            self.depth_map = DepthMap()

        print('')
