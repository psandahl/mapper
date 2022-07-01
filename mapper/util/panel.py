import cv2 as cv
import numpy as np

import random

import mapper.vision.image as im


class Panel():
    def __init__(self):
        self.pose_matches_win = 'pose_matches'

        self.pose_matches_image = None
        cv.namedWindow('pose_matches')

    def set_caption(self, caption: str) -> None:
        cv.setWindowTitle(self.pose_matches_win, f'pos matches: {caption}')

    def set_pose_matches(self,
                         train_image: np.ndarray,
                         train_points: np.ndarray,
                         train_pose_points: np.ndarray,
                         query_image: np.ndarray,
                         query_points: np.ndarray,
                         query_pose_points: np.ndarray) -> None:
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

        for index, train_point in enumerate(train_pose_points):
            query_point = query_pose_points[index]
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))

            cv.drawMarker(train_image_vis, im.to_cv_point(
                train_point), color=color)
            cv.drawMarker(query_image_vis, im.to_cv_point(
                query_point), color=color)

        self.pose_matches_image = np.hstack(
            (train_image_vis, query_image_vis))

    def update(self):
        if not self.pose_matches_image is None:
            cv.imshow(self.pose_matches_win, self.pose_matches_image)

    def destroy_window(self) -> None:
        cv.destroyWindow(self.pose_matches_win)
