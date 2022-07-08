import numpy as np
import cv2 as cv

import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.transform as trf


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
            f'landmark hash grid. From {len(landmarks)} landmarks, {cnt} are used')

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


def landmark_pose_estimation(landmarks: list, descriptor_pair: tuple,
                             intrinsic_mat: np.array, pose: np.ndarray,
                             image: np.ndarray) -> None:
    assert isinstance(landmarks, list)
    assert isinstance(descriptor_pair, tuple)
    assert len(descriptor_pair) == 2
    assert isinstance(intrinsic_mat, np.ndarray)
    assert intrinsic_mat.shape == (3, 3)
    assert isinstance(pose, np.ndarray)
    assert pose.shape == (3, 4)
    assert im.is_image(image)
    assert im.num_channels(image) == 1

    hash_grid = LandmarkHashGrid(
        landmarks, intrinsic_mat, pose, im.image_size(image))
