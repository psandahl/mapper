import cv2 as cv
import numpy as np

import mapper.vision.matrix as mat


def remap_frame(R: np.ndarray, map: np.ndarray) -> np.ndarray:
    """
    Remap a rotation frame, e.g. ECEF, to be a camera frame within
    ECEF. If no mapping shall be made, e.g. camera and world frames
    have the same axes, map can be set to identity.

    Parameters:
        R: 3x3 rotation matrix.
        map: 3x3 rotation matrix.
    """
    assert isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    assert isinstance(map, np.ndarray)
    assert map.shape == (3, 3)

    return R @ map.T


def world_to_camera_mat(ext: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a world coordinate to a camera coordinate using an extrinsic matrix.

    Parameters:
        ext: The extrinsic matrix.
        xyz: The world coordinate.

    Returns:
        The camera coordinate.
    """
    assert isinstance(ext, np.ndarray)
    assert ext.shape == (3, 4)
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    xyz_h = np.append(xyz, 1.0)
    return ext @ xyz_h


def camera_to_world_mat(ext: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a camera coordinate to a world coordinate using an extrinsic matrix.

    Parameters:
        ext: The extrinsic matrix.
        xyz: The camera coordinate.

    Returns:
        The world coordinate.
    """
    assert isinstance(ext, np.ndarray)
    assert ext.shape == (3, 4)
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    # Decomp gives camera to world from start.
    R, t = mat.decomp_extrinsic_matrix(ext)

    return R @ xyz + t


def world_to_camera_rtvec(rvec: np.ndarray, tvec: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a world coordinate to a camera coordinate using rvec and tvec.

    Parameters:
        rvec: Rotation vector.
        tvec: Translation vector.        
        xyz: The world coordinate.

    Returns:
        The camera coordinate.
    """
    assert isinstance(rvec, np.ndarray)
    assert len(rvec) == 3
    assert isinstance(tvec, np.ndarray)
    assert len(tvec) == 3
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    R, _ = cv.Rodrigues(rvec)

    return R @ xyz + tvec


def camera_to_world_rtvec(rvec: np.ndarray, tvec: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a camera coordinate to a world coordinate using rvec and tvec.

    Parameters:
        rvec: Rotation vector.
        tvec: Translation vector.        
        xyz: The camera coordinate.

    Returns:
        The world coordinate.
    """
    assert isinstance(rvec, np.ndarray)
    assert len(xyz) == 3
    assert isinstance(tvec, np.ndarray)
    assert len(tvec) == 3
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    R, _ = cv.Rodrigues(rvec)

    # rtvec always go from world to camera - so invert!
    return R.T @ xyz + R.T @ -tvec


def change_pose(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Change the pose/base by applying the delta to the pose.

    Parameters:
        pose: The pose to change.
        delta: The pose to apply.

    Returns:
        The new pose.
    """
    assert isinstance(pose, np.ndarray)
    assert pose.shape == (3, 4)
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (3, 4)

    new_pose = mat.homogeneous_matrix(
        pose) @ np.linalg.inv(mat.homogeneous_matrix(delta))

    return mat.decomp_homogeneous_matrix(new_pose)


def project_point(projection_mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Project a world coordinate to image.

    Parameters:
        projection_mat: A 3x4 projection matrix.
        xyz: The world point.

    Returns:
        An image point.
    """
    assert isinstance(projection_mat, np.ndarray)
    assert projection_mat.shape == (3, 4)
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    xyz_h = np.append(xyz, 1.0)
    px = projection_mat @ xyz_h
    px /= px[2]

    return px[:2]
