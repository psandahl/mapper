import numpy as np
import cv2 as cv

import mapper.matrix as mat


def relative_pose(train: np.ndarray, query: np.ndarray, E: np.ndarray,
                  intrinsic_matrix: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, tuple]:
    """
    Compute the relative pose from train to query. Expressed in train's
    frame. This estimation cannot estimate scale, default all translations
    are unit size.

    Parameters:
        train: Image points for the train image.
        query: Image points for the query image.
        E: Essential matrix between the images.
        intrinsic_matrix: An intrinsic matrix valid for E and both images.
        scale: Override scaling for translation.

    Returns:
        Tuple (translation vector, (yaw, pitch, roll)).
    """
    assert isinstance(
        train, np.ndarray), 'train is assumed to be an array'
    assert isinstance(
        query, np.ndarray), 'query is assumed to be an array'
    assert len(train) == len(query), 'train and query must be same size'
    assert isinstance(
        E, np.ndarray), 'E matrix is assumed to be a matrix'
    assert E.shape == (
        3, 3), 'E matrix is assumed to be 3x3'
    assert isinstance(
        intrinsic_matrix, np.ndarray), 'intrinsic matrix is assumed to be a matrix'
    assert intrinsic_matrix.shape == (
        3, 3), 'intrinsic matrix is assumed to be 3x3'

    # The recoverPose function will, if called the intuitive way, return
    # the pose from query to train, in query's frame. But in this case
    # from train to query, in train's frame is much more convenient. That's
    # why the call looks like it does.
    retval, R, t, mask = cv.recoverPose(E.T, query, train, intrinsic_matrix)

    ypr = mat.decomp_ypr_matrix_yxz(R)

    return (t.flatten() * scale, ypr)
