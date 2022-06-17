from typing import Tuple
from xmlrpc.client import Boolean
import cv2 as cv
import numpy as np

import math


def is_image(value: any) -> Boolean:
    """
    Check if something is an image.

    Parameters:
        image: Value to check if image.

    Returns:
        True if value is image, False otherwise.
    """
    return isinstance(value, np.ndarray) and (len(value.shape) == 2 or len(value.shape) == 3)


def num_channels(image: cv.Mat) -> int:
    """
    Get the number of channels for an image.

    Parameters:
        image: Image to check the number of channels for.

    Returns:
        The number of channels.
    """
    assert is_image(image), 'Argument is supposed to be an image'
    if len(image.shape) == 3:
        h, w, c = image.shape
        return c
    else:
        return 1


def image_size(image: cv.Mat) -> tuple:
    """
    Get the size of an image.

    Parameters:
        image: Image to check for size.

    Returns:
        Size in pixels as (width, height).
    """
    assert is_image(image), 'Argument is supposed to be an image'

    # Shape is of len 3 or 2 for images.
    if len(image.shape) == 3:
        h, w, c = image.shape
        return (w, h)
    else:
        h, w = image.shape
        return (w, h)


def read_image_bgr(path: str) -> cv.Mat:
    """
    Read an image, forced to be in BGR encoding.

    Parameters:
        path: Path to the image.

    Returns:
        The image, or None if not found.
    """
    return cv.imread(path, cv.IMREAD_COLOR)


def gray_convert(image: cv.Mat) -> cv.Mat:
    """
    Create a gray scale copy of an image.

    Parameters:
        image: Image to convert.

    Returns:
        The gray image.
    """
    assert is_image(image), 'Argument is supposed to be an image'

    if num_channels(image) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image.copy()


def visualization_image(image: cv.Mat) -> cv.Mat:
    """
    Create a 3-channel copy of an image, suitable for presentation visualizations.

    Parameters:
        image: Image to convert.

    Returns:
        The presentation image.
    """
    assert is_image(image), 'Argument is supposed to be an image'

    if num_channels(image) == 3:
        return image.copy()
    else:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def generate_features(image: cv.Mat) -> np.ndarray:
    """
    Generate corner features in the given image.

    Parameters:
        image: Image to generate features for.

    Returns:
        Numpy array with corner features.
    """
    assert is_image(image), 'Argument is supposed to be an image'
    assert num_channels(image) == 1, 'Image is supposed to be single channel'

    max_corners = 2500
    quality_level = 0.01

    w, h = image_size(image)

    # Max corners and image size determines the min distance between features.
    min_distance = min(w, h) / math.sqrt(max_corners)

    features = cv.goodFeaturesToTrack(
        image, max_corners, quality_level, min_distance)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 20, 0.05)

    print(
        f'generate_features() request/response ratio={len(features) / max_corners:.2f}%')

    return cv.cornerSubPix(image, features, (3, 3), (-1, -1), criteria)


def draw_features(image: cv.Mat, features: np.ndarray, color: tuple = (0, 255, 0)) -> any:
    """
    Show features as circles in the given image.

    Parameters:
        image: The visualization image.
        feature: The features.
        color: The color for the circles to be drawn.
    """
    assert is_image(image), 'Argument is supposed to be an image'
    assert num_channels(image) == 3, 'Image is supposed to have three channels'

    for feature in features:
        flat_feature = feature.flatten()
        center = (int(round(flat_feature[0])), int(round(flat_feature[1])))

        cv.circle(image, center, 3, color)
