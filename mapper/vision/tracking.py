import numpy as np
from scipy.optimize import least_squares
import cv2 as cv

import functools
import itertools

import mapper.vision.image as im
import mapper.vision.keypoint as kpt
import mapper.vision.matrix as mat
import mapper.vision.transform as trf
import mapper.vision.utils as utils


class LandmarkHashGrid():
    def __init__(self,
                 landmarks: list,
                 intrinsic_mat: np.ndarray,
                 pose: np.ndarray,
                 image_size: tuple,
                 grid: tuple = (10, 10)) -> None:
        """
        Create a hash grid from a list of landmarks. Landmarks
        will be projected to determine if they're infront of
        camera and within the image.
        """
        self.image_size = image_size
        self.grid = grid
        self.step = np.ceil(np.divide(image_size, grid)).astype(int)

        # The grid is a list of lists.
        num_grid_elements = grid[0] * grid[1]
        self.hash_grid = [list() for i in range(0, num_grid_elements)]

        R, t = mat.decomp_pose_matrix(pose)
        extrinsic_mat = mat.extrinsic_matrix(R, t)
        projection_mat = mat.projection_matrix(intrinsic_mat,
                                               extrinsic_mat)

        # Populate the hash grid with useful points.
        cnt = 0
        for landmark in landmarks:
            # Project the landmark into the current image if it's
            # infront of the camera.
            if trf.infront_of_camera(extrinsic_mat, landmark.get_xyz()):
                px = trf.project_point(projection_mat, landmark.get_xyz())
                # Query for a matching grid index ...
                grid_pos = self.px_to_grid_pos(px)
                if not grid_pos is None:
                    cnt += 1
                    # If found, insert the pixel and a reference to the landmark.
                    grid_index = self.grid_pos_to_grid_index(grid_pos)
                    self.hash_grid[grid_index].append((px, landmark))

        print(
            f'landmark hash grid. From {len(landmarks)} landmarks, {cnt} are stored')

    def get_landmarks(self, circle: tuple) -> list:
        """
        Get a set of landmarks matching the circle. Matches can be
        outside the radius due to grid cell configuration though.
        """
        center, radius = circle

        # Should only be called with a circle center within the target image.
        if not self.within_image(center):
            print('Warning: get_landmarks called with circle center outside image')
            return list()

        # Get the upper left and lower right pixels from radius.
        ul_px = np.array(center) - radius
        lr_px = np.array(center) + radius

        # Make sure that pixel boundaries are aligned with image boundaries.
        ul_px = np.clip(ul_px, (0, 0), np.array(self.image_size) - 1)
        lr_px = np.clip(lr_px, (0, 0), np.array(self.image_size) - 1)

        ul_grid_pos = self.px_to_grid_pos(ul_px)
        lr_grid_pos = self.px_to_grid_pos(lr_px)

        assert not ul_grid_pos is None
        assert not lr_grid_pos is None

        min_col, min_row = ul_grid_pos
        max_col, max_row = lr_grid_pos

        # Now iterate the cells identified by the boundaries, and append
        # the cell's index if the cell is intersecting with the circle.
        indices = list()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                grid_pos = (col, row)
                if utils.circle_rect_collide(circle, self.grid_pos_to_rect(grid_pos)):
                    indices.append(self.grid_pos_to_grid_index(grid_pos))

        # Fetch the data from the cell indicies.
        matches = list()
        for index in indices:
            assert index < len(self.hash_grid)
            matches.append(self.hash_grid[index])

        return list(itertools.chain(*matches))

    def within_image(self, px) -> bool:
        """Check if pixel is whithin image."""
        u, v = px
        w, h = self.image_size

        return u >= 0 and v >= 0 and u < w and v < h

    def px_to_grid_pos(self, px) -> any:
        """Transform pixel to grid position (col, row)."""
        if self.within_image(px):
            u, v = px
            step_w, step_h = self.step

            col = u // step_w
            row = v // step_h

            return (col, row)
        else:
            return None

    def grid_pos_to_grid_index(self, grid_pos: tuple) -> int:
        """Transform grid position (col, row) to linear grid index."""
        w, _ = self.grid
        col, row = grid_pos

        return int(row * w + col)

    def grid_pos_to_rect(self, grid_pos: tuple) -> tuple:
        """Get a rectangle (min, max) describing the area for the grid_pos."""
        step_w, step_h = self.step
        col, row = grid_pos

        min_x = col * step_w
        min_y = row * step_h

        max_x = min_x + step_w - 1
        max_y = min_y + step_h - 1

        return ((min_x, min_y), (max_x, max_y))


def visual_pose_prediction(match: dict, intrinsic_mat: np.ndarray, scale: float = 1.0) -> tuple:
    """
    From matched keypoints, do a pose predition.

    Parameters:
        match: Dictionary with matched keypoints and descriptors.
        instrinsic_matrix: The intrinsic matrix to be used 
        (assume the same matrix for train and query).

    Returns:
        A tuple (pose matrix describing base change from train to query, inlier matches).
    """
    assert isinstance(match, dict)
    assert isinstance(intrinsic_mat, np.ndarray)
    assert intrinsic_mat.shape == (3, 3)

    train = cv.KeyPoint_convert(match['train_keypoints'])
    query = cv.KeyPoint_convert(match['query_keypoints'])

    E, inliers = cv.findEssentialMat(np.array(train),
                                     np.array(query),
                                     intrinsic_mat,
                                     cv.RANSAC, 0.999, 0.1)
    inliers = inliers.flatten()

    kpt0 = list()
    desc0 = list()
    kpt1 = list()
    desc1 = list()
    for index, value in enumerate(inliers):
        if value == 1:
            kpt0.append(match['train_keypoints'][index])
            desc0.append(match['train_descriptors'][index])
            kpt1.append(match['query_keypoints'][index])
            desc1.append(match['query_descriptors'][index])

    match1 = dict()
    match1['train_keypoints'] = kpt0
    match1['train_descriptors'] = desc0
    match1['train_id'] = match['train_id']
    match1['query_keypoints'] = kpt1
    match1['query_descriptors'] = desc1
    match1['query_id'] = match['query_id']

    _, R, t, _ = cv.recoverPose(E, cv.KeyPoint_convert(kpt0),
                                cv.KeyPoint_convert(kpt1), intrinsic_mat)

    return (mat.pose_matrix(R, t.flatten() * scale), match1)


def landmark_pose_estimation(frame_id: int,
                             landmarks: list, descriptor_pair: tuple,
                             intrinsic_mat: np.array, pose: np.ndarray,
                             image: np.ndarray,
                             radius: int = 60) -> None:
    assert isinstance(landmarks, list)
    assert isinstance(descriptor_pair, tuple)
    assert len(descriptor_pair) == 2
    assert isinstance(intrinsic_mat, np.ndarray)
    assert intrinsic_mat.shape == (3, 3)
    assert isinstance(pose, np.ndarray)
    assert pose.shape == (3, 4)
    assert im.is_image(image)
    assert im.num_channels(image) == 1

    hamming_threshold = 60

    sq_radius = radius ** 2

    hash_grid = LandmarkHashGrid(
        landmarks, intrinsic_mat, pose, im.image_size(image))

    image_points = list()
    world_points = list()

    # Match the feature descriptors with landmarks in the same image region.
    features, descriptors = descriptor_pair
    for feature_index, feature in enumerate(features):
        descriptor = descriptors[feature_index]
        feature_px = np.array(feature.pt)

        # Annotate landmarks with hamming distance to feature.
        annotated_landmarks = list()
        landmarks = hash_grid.get_landmarks(
            (im.to_cv_point(feature_px), radius))
        for landmark_px, landmark in landmarks:

            # As the landmarks from the hash grid is cell based,
            # check the radius.
            sq_distance = np.square(landmark_px - feature_px).sum()
            if sq_distance < sq_radius:
                hamming_distance = kpt.hamming_distance(descriptor,
                                                        landmark.get_descriptor())
                if hamming_distance < hamming_threshold:
                    annotated_landmarks.append((hamming_distance, landmark))

        # Sort the annotated landmarks and filter using Lowe's ratio.
        annotated_landmarks.sort(key=lambda tup: tup[0])
        if len(annotated_landmarks) > 1:
            hamming_0, landmark_0 = annotated_landmarks[0]
            hamming_1, _ = annotated_landmarks[1]

            if hamming_0 < hamming_1 * 0.8:
                image_points.append(feature_px)
                world_points.append(landmark_0.get_xyz())

                landmark.mark_use(frame_id)

    print(
        f'landmark pose estimation. Frame id={frame_id}. Selected point pairs={len(image_points)}')
    if len(image_points) > 8:
        f = functools.partial(trf.project_points_opt_6dof,
                              image_points, world_points, intrinsic_mat, pose)
        result = least_squares(f, np.zeros(6), method='lm')
        if result.success:
            print(f'adjustment={result.x}')
