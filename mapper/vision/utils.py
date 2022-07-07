import numpy as np

import math


def fov_from_focal_length(focal_length: float, media_size: float) -> float:
    """
    Compute the field of view given a focal length and 
    a media size (in same units).

    Parameters:
        focal_length: The focal length.
        media_size: The size of the sensor/image media.

    Returns:
        The field of view in degrees.
    """
    fov = math.atan2(media_size / 2.0, focal_length) * 2.0

    return math.degrees(fov)


def focal_length_from_fov(fov: float, media_size: float) -> float:
    """
    Compute the focal length from field of view and media size.

    Parameters:
        fov: The field of view in degrees.
        media_size: The size of the sensor/image media.

    Returns:
        The focal length in the same unit as for the media size.
    """
    half_fov = math.radians(fov) / 2.0
    half_size = media_size / 2.0

    return half_size / math.tan(half_fov)


def matching_fov(fov: float, aspect_ratio: float) -> float:
    """
    Given a field of view and an aspect ratio, give back the matching
    field of view for the 'other' side.

    Parameters:
        fov: Field of view in degrees.

    Returns:
        The other field of view in degrees.
    """
    half_side = math.tan(math.radians(fov / 2.0))
    half_side *= aspect_ratio

    return math.degrees(math.atan2(half_side, 1.0) * 2.0)


def aspect_ratio(size: tuple) -> float:
    """
    Compute aspect ratio.

    Parameters:
        size: Tuple (width, height).

    Returns:
        The aspect ratio.
    """
    assert isinstance(size, tuple), 'Argument is assumed to be a tuple'
    assert len(size) == 2, 'Argument is assumed to have two elements'
    return size[0] / size[1]


def circle_rect_collide(circle: tuple, rect: tuple) -> bool:
    """
    Check whether a circle and a rectangle collides.

    Parameters:
        circle: Tuple ((x, y), radius).
        rect: Tuple ((min x, min y), (max x, max y))

    Returns:
        True if collision.
    """
    xy, radius = circle
    min, max = rect

    closest_point = np.clip(xy, min, max)

    return np.linalg.norm(closest_point - xy) < radius
