import cv2 as cv
import numpy as np

import math
import random

import mapper.vision.epipolar as epi
import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.utils as utils


class Panel():
    def __init__(self, data_extent: tuple):
        min, max, mid = data_extent

        self.pose_matches_win = 'pose_matches'
        self.track_win = 'track'

        # The pose matches image is initially not created. Will be
        # created at first set.
        self.pose_matches_image = None

        # Create the track image. It will incrementally be filled with
        # content.
        track_image_width = 1024
        track_image_height = 768

        self.track_image = np.zeros(
            (track_image_height, track_image_width, 3), dtype=np.uint8)
        #self.track_image[:, :, :] = 0

        # Setup camera for the track view.
        h_fov = 60
        v_fov = utils.matching_fov(h_fov,
                                   1 / utils.aspect_ratio((track_image_width,
                                                           track_image_height)))
        self.track_image_intrinsic_matrix = mat.ideal_intrinsic_matrix(
            (h_fov, v_fov), (track_image_width, track_image_height))

        # Place camera at min_z, max_x, looking at mid.
        min_x, min_z = min
        max_x, max_z = max
        mid_x, mid_z = mid

        eye = np.array([max_x * 2, -100.0, min_z * 2])
        at = np.array([mid_x, 0.0, mid_z])
        down = np.array([0.0, 1.0, 0.0])

        R, t = mat.look_at_yxz(eye, at, down)
        rvec, tvec = mat.extrinsic_rtvec(R, t)

        self.track_image_rvec = rvec
        self.track_image_tvec = tvec

        # Draw lines to mark the data extent.
        self.draw_track_extent(data_extent)

        cv.namedWindow(self.pose_matches_win)
        cv.namedWindow(self.track_win)

    def set_caption(self, caption: str) -> None:
        cv.setWindowTitle(self.pose_matches_win, f'pos matches: {caption}')
        cv.setWindowTitle(self.track_win, f'track: {caption}')

    def set_pose_matches(self,
                         train_image: np.ndarray,
                         train_points: np.ndarray,
                         train_pose_points: np.ndarray,
                         query_image: np.ndarray,
                         query_points: np.ndarray,
                         query_pose_points: np.ndarray,
                         F: np.ndarray = None) -> None:
        assert im.is_image(
            train_image), 'Train image is assumed to be valid image'
        assert im.is_image(
            query_image), 'Query image is assumed to be valid image'
        assert len(train_pose_points) == len(
            query_pose_points), 'Pose points must have same length'

        train_image_vis = im.visualization_image(train_image)
        query_image_vis = im.visualization_image(query_image)

        im.draw_features(train_image_vis, train_points)
        im.draw_features(query_image_vis, query_points)

        image_size = im.image_size(query_image_vis)

        for index, train_point in enumerate(train_pose_points):
            query_point = query_pose_points[index]
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))

            cv.drawMarker(train_image_vis, im.to_cv_point(
                train_point), color=color)
            cv.drawMarker(query_image_vis, im.to_cv_point(
                query_point), color=color)

            if not F is None:
                line = epi.epipolar_line(F, train_point)
                p0, p1 = epi.plot_line(line, image_size)
                cv.line(query_image_vis, p0, p1, color=color)

        self.pose_matches_image = np.hstack(
            (train_image_vis, query_image_vis))

    def update(self) -> None:
        if not self.pose_matches_image is None:
            cv.imshow(self.pose_matches_win, self.pose_matches_image)

        cv.imshow(self.track_win, self.track_image)

    def destroy_window(self) -> None:
        cv.destroyWindow(self.pose_matches_win)
        cv.destroyWindow(self.track_win)

    def draw_track_extent(self, data_extent: tuple) -> None:
        min, max, _ = data_extent
        min_x, min_z = min
        max_x, max_z = max

        c00 = np.array([min_x, 0.0, min_z])
        c01 = np.array([min_x, 0.0, max_z])
        c11 = np.array([max_x, 0.0, max_z])
        c10 = np.array([max_x, 0.0, min_z])

        corners, _ = cv.projectPoints(np.array([c00, c01, c11, c10]),
                                      self.track_image_rvec,
                                      self.track_image_tvec,
                                      self.track_image_intrinsic_matrix,
                                      None)

        corners = [im.to_cv_point(corner.flatten()) for corner in corners]
        cv.line(self.track_image,
                corners[0], corners[1], color=(0, 255, 0))
        cv.line(self.track_image,
                corners[1], corners[2], color=(0, 255, 0))
        cv.line(self.track_image,
                corners[2], corners[3], color=(0, 255, 0))
        cv.line(self.track_image,
                corners[3], corners[0], color=(0, 255, 0))

    def add_axes(self, pose_matrix: np.ndarray) -> None:
        R, t = mat.decomp_pose_matrix(pose_matrix)

        center_c = np.array([0.0, 0.0, 0.0])
        front_c = np.array([0.0, 0.0, 1.0])
        right_c = np.array([1.0, 0.0, 0.0])
        down_c = np.array([0.0, 1.0, 0.0])
        points_c = [center_c, front_c, right_c, down_c]

        points_w = [R @ point + t for point in points_c]

        points, _ = cv.projectPoints(np.array(points_w),
                                     self.track_image_rvec,
                                     self.track_image_tvec,
                                     self.track_image_intrinsic_matrix,
                                     None)
        points = [im.to_cv_point(point.flatten()) for point in points]
        cv.line(self.track_image,
                points[0], points[1], color=(0, 0, 255))
        cv.line(self.track_image,
                points[0], points[2], color=(0, 255, 0))
        cv.line(self.track_image,
                points[0], points[3], color=(255, 0, 0))

    def add_camera(self, pose_matrix: np.ndarray, color: tuple = (0, 255, 0)) -> None:
        R, t = mat.decomp_pose_matrix(pose_matrix)

        # Five points describing the frustum shape in camera space.
        half_size = math.tan(math.radians(40) / 2.0)
        points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-half_size, half_size, 1.0]),
            np.array([-half_size, -half_size, 1.0]),
            np.array([half_size, -half_size, 1.0]),
            np.array([half_size, half_size, 1.0])
        ]

        # To world space.
        points = [R @ point + t for point in points]

        # To image space.
        points, _ = cv.projectPoints(np.array(points),
                                     self.track_image_rvec,
                                     self.track_image_tvec,
                                     self.track_image_intrinsic_matrix,
                                     None)
        points = [im.to_cv_point(point.flatten()) for point in points]

        cv.line(self.track_image, points[0], points[1], color)
        cv.line(self.track_image, points[0], points[2], color)
        cv.line(self.track_image, points[0], points[3], color)
        cv.line(self.track_image, points[0], points[4], color)

        cv.line(self.track_image, points[1], points[2], color)
        cv.line(self.track_image, points[2], points[3], color)
        cv.line(self.track_image, points[3], points[4], color)
        cv.line(self.track_image, points[4], points[1], color)

    def add_landmarks(self, landmarks: list) -> None:
        for landmark in landmarks:
            px, _ = cv.projectPoints(np.array(landmark.get_xyz()),
                                     self.track_image_rvec,
                                     self.track_image_tvec,
                                     self.track_image_intrinsic_matrix,
                                     None)
            px = np.round(px.flatten())
            if im.within_image(self.track_image, px):
                intensity = round(landmark.get_intensity())
                u, v = px.astype(int)
                self.track_image[v, u] = (
                    intensity, intensity, intensity)
