import numpy as np

import sys

import mapper.vision.matrix as mat


def last_in(xs: list) -> any:
    """
    Get the last element in a list.

    Parameters:
        xs: The list.

    Returns:
        The last value.
    """
    assert isinstance(xs, list), 'Argument is assumed to be a list'
    assert len(xs) > 0, 'List must not be empty'

    return xs[len(xs) - 1]


def read_2d_box_from_3x4_matrices(filename: str) -> tuple():
    """
    Get a simple 2d box from a list of 3x4 pose matrices.

    Parameters:
        filename: Name of the textfile to read.

    Returns:
        A tuple ((min_x, min_z), (max_x, max_z), (mid_x, mid_z)).
    """
    poses = read_3x4_matrices(filename)

    min_x = +1e09
    min_z = +1e09
    max_x = -1e09
    max_z = -1e09

    for pose in poses:
        _, t = mat.decomp_pose_matrix(pose)
        x, _, z = t

        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_z = min(z, min_z)
        max_z = max(z, max_z)

    mid_x = min_x + (max_x - min_x) / 2
    mid_z = min_z + (max_z - min_z) / 2

    return ((min_x, min_z), (max_x, max_z), (mid_x, mid_z))


def read_3x4_matrices(filename: str) -> list():
    """
    Read a list of 3x4 matrices from the given file.

    Parameters:
        filename: Name of the textfile to read.

    Returns:
        A list of 3x4 matrices.
    """
    matrices = list()
    with open(filename, 'r') as f:
        for line in f.readlines():
            array = np.fromstring(line, dtype=np.float64, sep=' ')
            if len(array) == 12:
                matrices.append(array.reshape(3, 4))
            else:
                print(
                    f'Warning: Line did not contain 12 values in file={filename}')

    return matrices
