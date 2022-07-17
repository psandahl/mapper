import numpy as np

import math


def epipolar_line(F: np.ndarray, px: any) -> np.ndarray:
    """
    Formulate an epipolar line in the 'right' image for the
    pixel in the 'left' image. F is transposable.

    Parameters:
        F: Fundamental matrix.
        px: A pixel coordinate.

    Returns:
        A tuple (a, b, c) describing a line ax + by + c = 0.
    """
    assert isinstance(F, np.ndarray)
    assert F.shape == (3, 3)
    assert len(px) == 2

    px_h = np.append(px, 1.0)
    a, b, c = F @ px_h

    # Normalize line such that a ** 2 + b ** 2 == 1.
    norm = math.sqrt(a ** 2 + b ** 2)

    return np.array([a, b, c]) / norm


def line_y(line: np.ndarray, x: float) -> float:
    """
    Get the y value for the given x value on the line.

    Parameters:
        line: A line.
        x: The x value.

    Returns:
        The y value.
    """
    assert isinstance(line, np.ndarray)
    assert line.shape == (3,)

    # ax + by + c = 0
    a, b, c = line

    return -(a * x + c) / b


def line_x(line: np.ndarray, y: float) -> float:
    """
    Get the x value for the given y value on the line.

    Parameters:
        line: A line.
        y: The y value.

    Returns:
        The x value.
    """
    assert isinstance(line, np.ndarray)
    assert line.shape == (3,)

    # ax + by + c = 0
    a, b, c = line

    return -(b * y + c) / a


def clamp_point_to_image(line: np.ndarray, size: any, x: float) -> tuple:
    """
    Clamp a point to the border of an image. Selection is x.

    Parameters:
        line: A line.
        size: The size of the image.
        x: The ideal x value for the selection.

    Returns:
        A tuple (x, y) clamped to the image border.
    """
    assert isinstance(line, np.ndarray)
    assert line.shape == (3,)
    assert len(size) == 2
    _, h = size

    # Test: see if a plotted y fits along the image side.
    y = line_y(line, x)
    if y < 0.0:
        # y must be clamped to zero.
        x = line_x(line, 0)
        return x, 0.0
    elif y > h - 1.0:
        # y must be clamped to h - 1
        x = line_x(line, h - 1)
        return x, h - 1
    else:
        return x, y


def clamp_line_to_image(line: np.ndarray, size: tuple) -> tuple:
    """
    Clamp a line to the border of an image.
    """
    w, _ = size
    return clamp_point_to_image(line, size, 0), clamp_point_to_image(line, size, w - 1)


def plot_line(line: np.ndarray, size: tuple) -> tuple:
    """
    Clamp a line to the border of an image. Force integer typing (for OpenCV plot).
    """
    start, end = clamp_line_to_image(line, size)

    return np.round(start).astype(int), np.round(end).astype(int)
