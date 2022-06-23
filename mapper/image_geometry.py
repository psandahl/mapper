import cv2 as cv
import numpy as np

import mapper.image as im


def dense_optical_flow(image0: cv.Mat, image1: cv.Mat) -> cv.Mat:
    """
    Calculate the dense optical flow between two equal sized, one channel, images.

    Parameters:
        image0: First image to optical flow.
        image1: Second image to optical flow.

    Returns:
        A two channel floating point image with flow from image0 to image1.
    """
    assert im.is_image(image0), 'Image0 is assumed to be an image'
    assert im.num_channels(
        image0) == 1, 'Image0 is supposed to have one channel'
    assert im.is_image(image1), 'Image1 is assumed to be an image'
    assert im.num_channels(
        image1) == 1, 'Image1 is supposed to have one channel'
    assert im.image_size(image0) == im.image_size(
        image1), 'Images are supposed to have same size'

    rows, cols = image0.shape
    flow = np.zeros(shape=(rows, cols, 2), dtype=np.float32)

    # MEDIUM, FAST, ULTRA_FAST are the creation options.
    dis = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(image0, image1, flow)


def matching_features_from_flow(flow: cv.Mat, features: np.ndarray) -> np.ndarray:
    """
    From a flow image and a set of image features compute target features.

    Parameters:
        flow: A flow image from image0 to image1.
        features: A set of features from image0.

    Returns:
        Features for image1.
    """
    assert im.is_image(flow), 'Argument is assumed to be an image'
    assert im.num_channels(
        flow) == 2, 'Argument is supposed to have two channels'
    assert isinstance(
        features, np.ndarray), 'Argument is supposed to be an array'

    targets = list()
    for feature in features:
        x, y = feature
        targets.append(feature + im.interpolate_pixel(flow, x, y))

    return np.array(targets)
