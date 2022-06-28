import numpy as np


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
