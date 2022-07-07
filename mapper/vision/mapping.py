import numpy as np
import cv2 as cv

import mapper.vision.image as im
import mapper.vision.keypoint as kp
import mapper.vision.matrix as mat
import mapper.vision.transform as trf


class Landmark():
    def __init__(self, xyz: np.ndarray, id: int, frame_0: dict, frame_1: dict):
        """Initialize a new landmark"""
        self.xyz = xyz
        self.frame_created = id
        self.frame_0 = frame_0
        self.frame_1 = frame_1
        self.use_count = 0
        self.latest_frame_used = id

    def get_xyz(self) -> np.ndarray:
        return self.xyz

    def get_intensity(self) -> float:
        return self.frame_0['intensity']

    def get_hamming_distance(self) -> int:
        """
        Get the hamming distance between the feature descriptors
        from which this landmark was triangulated.
        """
        return kp.hamming_distance(self.frame_0['descriptor'],
                                   self.frame_1['descriptor'])

    def get_intensity_distance(self) -> float:
        """
        Get the pixel intensity distance between the images
        from which this landmark was triangulated.
        """
        return abs(self.frame_0['intensity'] - self.frame_1['intensity'])


def sparse_mapping(frame_id: int,
                   pose_0: np.ndarray,
                   intrinsic_0: np.ndarray,
                   image_0: np.ndarray,
                   pose_1: np.ndarray,
                   intrinsic_1: np.ndarray,
                   image_1: np.ndarray,
                   match: dict,
                   landmarks: list) -> None:
    """
    Perform sparse mapping through triangulation.

    Parameters:
        frame_id: The current frame id.
        pose_0: The pose matrix for frame 0.
        instrinsic_0: The intrinsic matrix for frame 0.
        image_0: The image for frame 0.
        pose_1: The pose matrix for frame 1.
        instrinsic_1: The intrinsic matrix for frame 1.
        image_1: The image for frame 1.
        match: The match dictionary between frame 0 and frame 1.
        landmarks: Where to store new landmarks.

    Returns:
        None
    """
    assert isinstance(frame_id, int)
    assert frame_id > 0
    assert isinstance(pose_0, np.ndarray)
    assert pose_0.shape == (3, 4)
    assert isinstance(intrinsic_0, np.ndarray)
    assert intrinsic_0.shape == (3, 3)
    assert im.is_image(image_0)
    assert isinstance(pose_1, np.ndarray)
    assert pose_1.shape == (3, 4)
    assert isinstance(intrinsic_1, np.ndarray)
    assert intrinsic_1.shape == (3, 3)
    assert im.is_image(image_1)
    assert isinstance(match, dict)
    assert isinstance(landmarks, list)
    assert len(match['train_keypoints']) == len(match['query_keypoints'])

    train_id = match['train_id']
    query_id = match['query_id']
    num_input_points = len(match['train_keypoints'])

    print(
        f'sparse mapping for frame={frame_id} between 0={train_id}, 1={query_id}. {num_input_points} pts')

    R_0, t_0 = mat.decomp_pose_matrix(pose_0)
    extrinsic_0 = mat.extrinsic_matrix(R_0, t_0)
    projection_0 = mat.projection_matrix(intrinsic_0, extrinsic_0)

    R_1, t_1 = mat.decomp_pose_matrix(pose_1)
    extrinsic_1 = mat.extrinsic_matrix(R_1, t_1)
    projection_1 = mat.projection_matrix(intrinsic_1, extrinsic_1)

    img_points_0 = cv.KeyPoint_convert(match['train_keypoints'])
    img_points_1 = cv.KeyPoint_convert(match['query_keypoints'])

    world_points = cv.triangulatePoints(projection_0, projection_1,
                                        img_points_0.T, img_points_1.T)
    world_points = np.transpose(world_points[:3] / world_points[3])

    # Create landmarks from inliers.
    desc_0 = match['train_descriptors']
    desc_1 = match['query_descriptors']

    inliers = 0
    for index, xyz in enumerate(world_points):
        c_0 = trf.world_to_camera_mat(extrinsic_0, xyz)
        c_1 = trf.world_to_camera_mat(extrinsic_1, xyz)
        if c_0[2] > 0.0 and c_1[2] > 0.0:
            uv_0 = img_points_0[index]
            frame_0 = dict({
                'id': train_id,
                'uv': uv_0,
                'pose': pose_0,
                'intensity': im.interpolate_pixel(image_0, uv_0[0], uv_0[1]),
                'descriptor': desc_0[index]
            })

            uv_1 = img_points_1[index]
            frame_1 = dict({
                'id': query_id,
                'uv': uv_1,
                'pose': pose_1,
                'intensity': im.interpolate_pixel(image_1, uv_1[0], uv_1[1]),
                'descriptor': desc_1[index]
            })

            landmark = Landmark(xyz, frame_id, frame_0, frame_1)
            landmarks.append(landmark)

            inliers += 1

    print(f' mapping ratio={inliers}/{num_input_points}')
