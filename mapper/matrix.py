import cv2 as cv
from cv2 import HOUGH_GRADIENT
import numpy as np

import mapper.utils as utils


def ideal_intrinsic_matrix(fov: tuple, image_size: tuple) -> cv.Mat:
    """
    Compute an ideal intrinsic matrix (center at image center etc.).

    Parameters:
        fov: Tuple (horizontal fov, vertical fov) in degrees.
        image_size: Tuple (width, height) for the image (assuming pixels).

    Returns:
        A 3x3 intrinsic matrix.
    """
    assert isinstance(fov, tuple), "fov is assumed to be a tuple"
    assert len(fov) == 2, "fov shall have two elements"
    assert isinstance(image_size, tuple), "image_size is assumed to be a tuple"
    assert len(image_size) == 2, "image_size shall have two elements"

    h_fov, v_fov = fov
    w, h = image_size

    f_x = utils.focal_length_from_fov(h_fov, w)
    f_y = utils.focal_length_from_fov(v_fov, h)

    c_x = (w - 1) / 2.0
    c_y = (h - 1) / 2.0

    m = [f_x, 0.0, c_x, 0.0, f_y, c_y, 0.0, 0.0, 1.0]

    return np.array(m).reshape(3, 3)


def decomp_intrinsic_matrix(mat: cv.Mat) -> tuple:
    """
    Decompose the intrinsic matrix.

    Parameters:
        mat: A 3x3 intrinsic matrix.

    Returns:
        A nested tuple ((h_fov, vfov), (w, h)).
    """
    assert isinstance(mat, np.ndarray), 'Argument is assumed to be a matrix'
    assert mat.shape == (3, 3), 'Matrix is assumed to be 3x3'

    w = mat[0, 2] * 2.0 + 1
    h = mat[1, 2] * 2.0 + 1

    h_fov = utils.fov_from_focal_length(mat[0, 0], w)
    v_fov = utils.fov_from_focal_length(mat[1, 1], h)

    return ((h_fov, v_fov), (w, h))


def intrinsic_matrix_35mm_film(focal_length: float, image_size: tuple) -> cv.Mat:
    """
    Helper function to compute an intrinsic matrix for a 35mm file (e.g. iphone Exif).
    """
    assert isinstance(image_size, tuple), 'Argument is assumed to be a tuple'
    assert len(image_size) == 2, 'Argument is assumed to have two elements'

    aspect_ratio = utils.aspect_ratio(image_size)
    h_fov = utils.fov_from_focal_length(focal_length, 35)
    v_fov = utils.fov_from_focal_length(focal_length, 35 / aspect_ratio)

    return ideal_intrinsic_matrix((h_fov, v_fov), image_size)
