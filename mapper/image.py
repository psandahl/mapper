from typing import Tuple
from xmlrpc.client import Boolean
import cv2 as cv
import numpy as np


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
    Gray convert an image.

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
