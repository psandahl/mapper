import numpy as np

import mapper.matrix as mat


def world_to_camera(ext: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a world coordinate to a camera coordinate.

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


def camera_to_world(ext: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    R, t = mat.decomp_extrinsic_matrix(ext)
    return R.T @ xyz - R.T @ t


def self_from_extrinsic_matrix(ext: np.ndarray) -> np.ndarray:
    """
    Get the self position in world space from an extrinsic matrix.

    Parameters:
        ext: The extrinsic matrix.

    Returns:
        The world translation/position.
    """
    assert isinstance(ext, np.ndarray), 'Argument is assumed to be a matrix'
    assert ext.shape == (3, 4), 'Matrix is assumed to be 3x4'

    R, t = mat.decomp_extrinsic_matrix(ext)
    return R.T @ -t
