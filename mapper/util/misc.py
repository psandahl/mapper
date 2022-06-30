import numpy as np


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
