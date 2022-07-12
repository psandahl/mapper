import cv2 as cv
import numpy as np

import mapper.vision.image as im


def points_to_track(image: np.ndarray) -> np.ndarray:
    """
    Detect trackable points in the provided image.

    Parameters:
        image: Single channel image.

    Returns:
        Array with detected image points.
    """
    assert im.is_image(image)
    assert im.num_channels(image) == 1

    w, h = im.image_size(image)

    distance = 10
    max_corners = int(w / distance * h / distance)

    points = im.flatten_feature_array(
        cv.goodFeaturesToTrack(image, max_corners, 0.025, distance))

    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 20, 0.05)
    return cv.cornerSubPix(image, points, (3, 3), (-1, -1), criteria)


def sparse_optical_flow(image0: np.ndarray, image1: np.ndarray, points0: np.ndarray) -> tuple:
    """
    Calculate the sparse optical flow between the two images using 
    the provided points.

    Parameters:
        image0: First image to optical flow.
        image1: Second image to optical flow.
        points0: Points taken from the first image.

    Returns:
        A tuple (match0, match1) with mathing points.
    """
    assert im.is_image(image0)
    assert im.num_channels(image0) == 1
    assert im.is_image(image1)
    assert im.num_channels(image1) == 1
    assert isinstance(points0, np.ndarray)

    args = dict(winSize=(15, 15), maxLevel=4)

    p_0_1, st_0_1, _ = cv.calcOpticalFlowPyrLK(
        image0, image1, points0, None, **args)
    p_1_0, st_1_0, _ = cv.calcOpticalFlowPyrLK(
        image1, image0, p_0_1, None, **args)

    match0 = list()
    match1 = list()
    for index, point in enumerate(points0):
        status_ok = st_0_1[index] == 1 and st_1_0[index] == 1
        if status_ok and np.linalg.norm(point - p_1_0[index]) < 1.0:
            match0.append(point)
            match1.append(p_0_1[index])

    return np.array(match0), np.array(match1)


def dense_optical_flow(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Calculate the dense optical flow between two equal sized, one channel, images.

    Parameters:
        image0: First image to optical flow.
        image1: Second image to optical flow.

    Returns:
        A two channel floating point image with flow from image0 to image1.
    """
    assert im.is_image(image0)
    assert im.num_channels(image0) == 1
    assert im.is_image(image1)
    assert im.num_channels(image1) == 1
    assert im.image_size(image0) == im.image_size(image1)

    rows, cols = image0.shape
    flow = np.zeros(shape=(rows, cols, 2), dtype=np.float32)

    # MEDIUM, FAST, ULTRA_FAST are the creation options.
    dis = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(image0, image1, flow)


def matching_features_from_dense_flow(flow: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    From a flow image and a set of image features compute target features.

    Parameters:
        flow: A flow image from image0 to image1.
        features: A set of features from image0.

    Returns:
        Features for image1.
    """
    assert im.is_image(flow)
    assert im.num_channels(flow) == 2
    assert isinstance(features, np.ndarray)

    targets = list()
    for feature in features:
        x, y = feature
        targets.append(feature + im.interpolate_pixel(flow, x, y))

    return np.array(targets)
