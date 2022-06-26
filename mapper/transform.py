import numpy as np

import mapper.matrix as mat


def self_from_extrinsic_matrix(m: np.ndarray) -> np.ndarray:
    """
    Get the self position in world space from an extrinsic matrix.

    Parameters:
        m: The extrinsic matrix.

    Returns:
        The world translation/position.
    """
    assert isinstance(m, np.ndarray), 'Argument is assumed to be a matrix'
    assert m.shape == (3, 4), 'Matrix is assumed to be 3x4'

    R, t = mat.decomp_extrinsic_matrix(m)
    return R.T @ -t
