import numpy as np
import cv2 as cv

import mapper.vision.matrix as mat


def visual_pose_prediction(match: dict, intrinsic_matrix: np.ndarray, scale: float = 1.0) -> tuple:
    """
    From matched keypoints, do a pose predition.

    Parameters:
        match: Dictionary with matched keypoints and descriptors.
        instrinsic_matrix: The intrinsic matrix to be used 
        (assume the same matrix for train and query).

    Returns:
        A tuple (pose matrix describing base change from train to query, inlier matches).
    """
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

    return (mat.pose_matrix(R, t.flatten() * scale), match1)
