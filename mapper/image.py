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

    features = flatten_feature_array(
        cv.goodFeaturesToTrack(image, max_corners,
                               quality_level, min_distance))
    print(
        f'generate_features() response/request ratio={len(features) / max_corners:.2f}%')

    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 20, 0.05)

    return cv.cornerSubPix(image, features, (3, 3), (-1, -1), criteria)


def draw_features(image: cv.Mat, features: np.ndarray, color: tuple = (0, 255, 0)) -> None:
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
        center = (int(round(feature[0])), int(round(feature[1])))

        cv.circle(image, center, 3, color)


def flatten_feature_array(xs: np.ndarray) -> np.ndarray:
    """Helper function to flatten an array of image points"""
    assert isinstance(xs, np.ndarray), 'Argument is supposed to be an array'

    if len(xs) > 0 and isinstance(xs[0][0], np.ndarray):
        return np.array([x.flatten() for x in xs])
    else:
        return xs


def dense_optical_flow(image0: cv.Mat, image1: cv.Mat) -> cv.Mat:
    """
    Calculate the dense optical flow between two equal sized, one channel, images.

    Parameters:
        image0: First image to optical flow.
        image1: Second image to optical flow.

    Returns:
        A two channel floating point image with flow from image0 to image1.
    """
    assert is_image(image0), 'Image0 is assumed to be an image'
    assert num_channels(image0) == 1, 'Image0 is supposed to have one channel'
    assert is_image(image1), 'Image1 is assumed to be an image'
    assert num_channels(image1) == 1, 'Image1 is supposed to have one channel'
    assert image_size(image0) == image_size(
        image1), 'Images are supposed to have same size'

    rows, cols = image0.shape
    flow = np.zeros(shape=(rows, cols, 2), dtype=np.float32)

    # MEDIUM, FAST, ULTRA_FAST are the creation options.
    dis = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_FAST)
    return dis.calc(image0, image1, flow)


def gray_flow_visualization_image(flow: cv.Mat) -> cv.Mat:
    """
    Create a gray scale vizualization image showing the magnitude of flow.

    Parameters:
        flow: A flow image.

    Returns:
        A normalized floating point image with magnitudes.
    """
    assert is_image(flow), 'Argument is assumed to be an image'
    assert num_channels(flow) == 2, 'Argument is supposed to have two channels'

    rows, cols, channels = flow.shape
    viz = np.zeros(shape=(rows, cols), dtype=np.float32)

    itr = np.nditer(viz, flags=['multi_index'])
    for px in itr:
        row, col = itr.multi_index
        viz[row, col] = np.linalg.norm(flow[row, col])

    return cv.normalize(viz, viz, 1.0, 0.0, cv.NORM_MINMAX)


def interpolate_pixel(image: cv.Mat, x: float, y: float) -> any:
    """
    Interpolate an image from floating point pixel coordinates.

    Parameters:
        image: The image to be read.
        x: x image coordinate.
        y: y image coordinate.

    Returns:
        The weighted image value.  
    """
    assert is_image(image), 'Argument is assumed to be an image'

    # Integer part.
    i_x = math.floor(x)
    i_y = math.floor(y)

    w, h = image_size(image)
    if i_x < w - 1 and i_y < h - 1:
        # Fractional part.
        f_x = x - i_x
        f_y = y - i_y

        w00 = (1.0 - f_x) * (1.0 - f_y)
        w10 = f_x * (1.0 - f_y)
        w01 = (1.0 - f_x) * f_y
        w11 = f_x * f_y

        px00 = image[i_y, i_x]
        px10 = image[i_y, i_x + 1]
        px01 = image[i_y + 1, i_x]
        px11 = image[i_y + 1, i_x + 1]

        return w00 * px00 + w10 * px10 + w01 * px01 + w11 * px11
    elif i_x < w and i_y < h:
        # No interpolation at the border.
        return image[i_y, i_x]
    else:
        return None
