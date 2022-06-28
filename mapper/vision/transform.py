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
    assert isinstance(R, np.ndarray), 'Argument is assumed to be a matrix'
    assert R.shape == (3, 3), 'Matrix is assumed to be 3x3'
    assert isinstance(map, np.ndarray), 'Argument is assumed to be a matrix'
    assert map.shape == (3, 3), 'Matrix is assumed to be 3x3'
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
    assert isinstance(ext, np.ndarray), 'Argument is assumed to be a matrix'
    assert ext.shape == (3, 4), 'Matrix is assumed to be 3x4'
    assert isinstance(xyz, np.ndarray), 'Argument is assumed to be an array'
    assert len(xyz) == 3, 'Array is assumed to be of length 3'

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
    assert isinstance(ext, np.ndarray), 'Argument is assumed to be a matrix'
    assert ext.shape == (3, 4), 'Matrix is assumed to be 3x4'
    assert isinstance(xyz, np.ndarray), 'Argument is assumed to be an array'
    assert len(xyz) == 3, 'Array is assumed to be of length 3'

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
    assert isinstance(rvec, np.ndarray), 'Argument is assumed to be an array'
    assert len(rvec) == 3, 'rvec is assumed to of length 3'
    assert isinstance(tvec, np.ndarray), 'Argument is assumed to be an array'
    assert len(tvec) == 3, 'tvec is assumed to of length 3'
    assert isinstance(xyz, np.ndarray), 'Argument is assumed to be an array'
    assert len(xyz) == 3, 'Array is assumed to be of length 3'

    R, jac = cv.Rodrigues(rvec)

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
    R, jac = cv.Rodrigues(rvec)

    # rtvec always go from world to camera - so invert!
    return R.T @ xyz + R.T @ -tvec
