import numpy as np
import cv2 as cv

import mapper.vision.image as im


def sparse_mapping(frame_id: int,
                   pose_0: np.ndarray,
                   intrinsic_0: np.ndarray,
                   image_0: np.ndarray,
                   pose_1: np.ndarray,
                   intrinsic_1: np.ndarray,
                   image_1: np.ndarray,
                   match: dict) -> None:
    assert isinstance(frame_id, int)
    assert frame_id > 0
    assert isinstance(pose_0, np.ndarray)
    assert pose_0.shape == (3, 4)
    assert isinstance(intrinsic_0, np.ndarray)
    assert intrinsic_0.shape == (3, 3)
    assert im.is_image(image_0)
    assert isinstance(pose_1, np.ndarray)
    assert pose_1.shape == (3, 4)
    assert isinstance(intrinsic_1, np.ndarray)
    assert intrinsic_1.shape == (3, 3)
    assert im.is_image(image_1)
    assert isinstance(match, dict)
