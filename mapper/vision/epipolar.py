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
    Get the y value for the x value on the line.

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


def plot_line(line: np.ndarray, x0: float, x1: float) -> tuple:
    """
    Get start and endpoints for a line. Truncated to ints to fit OpenCV.

    Parameters:
        line: A line.
        x0: First x value.
        x1: Last x value.

    Returns:
        Two tuples, with first and last point in line respectively.
    """
    y0 = line_y(line, x0)
    y1 = line_y(line, x1)

    return (int(x0), int(y0)), (int(x1), int(y1))
