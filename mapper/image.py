from xmlrpc.client import Boolean
import cv2 as cv
import numpy as np


def is_image(image: any) -> Boolean:
    return isinstance(image, np.ndarray)


def num_channels(image: cv.Mat) -> int:
    assert is_image(image), 'Argument is supposed to be an image'
    if len(image.shape) == 3:
        h, w, c = image.shape
        return c
    else:
        return 1


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
    assert is_image(image), 'Argument is supposed to be an image'

    if num_channels(image) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image.copy()
