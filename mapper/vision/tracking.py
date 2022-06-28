import numpy as np
import cv2 as cv

import mapper.vision.matrix as mat


def visual_pose_prediction(match: dict, intrinsic_matrix: np.ndarray, scale: float = 1.0):
    assert isinstance(match, dict), 'match is assumed to be a dictionary'
    assert isinstance(
        intrinsic_matrix, np.ndarray), 'intrinsic matrix is assumed to be a matrix'
    assert intrinsic_matrix.shape == (
        3, 3), 'intrinsic matrix is assumed to be 3x3'

    train = cv.KeyPoint_convert(match['train_keypoints'])
    query = cv.KeyPoint_convert(match['query_keypoints'])

    E, inliers = cv.findEssentialMat(np.array(train),
                                     np.array(query),
                                     intrinsic_matrix,
                                     cv.RANSAC, 0.999, 0.075)
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
    match1['query_keypoints'] = kpt1
    match1['query_descriptors'] = desc1

    _, R, t, _ = cv.recoverPose(E, cv.KeyPoint_convert(kpt0),
                                cv.KeyPoint_convert(kpt1), intrinsic_matrix)

    return (mat.pose_matrix(R, t.flatten()), match1)


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
