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


def mix(v0: float, w0: float, v1: float, w1: float) -> float:
    """
    Simple mix function to mix linearly between two values.

    Parameters:
        v0: First value.
        w0: Weight for the first value [0 - 1]
        v1: Second value.
        w1: Weight for the second value 1 - w0

    Returns:
        The mixed value.
    """
    return v0 * w0 + v1 * w1


def depth_to_bgr(depth: float, max_depth: float) -> tuple:
    """
    Simple function to map from a depth value to a b, g, r color tuple.
    """
    depth = max(0.0, min(depth, max_depth))

    b, g, r = 0, 0, 0
    if depth < max_depth / 2.0:
        green = depth / (max_depth / 2.0)
        r = int(round(mix(255.0, 1 - green, 0.0, green)))
        g = int(round(mix(255.0, green, 0.0, 1 - green)))
    else:
        blue = (depth - max_depth / 2.0) / (max_depth / 2.0)
        g = int(round(mix(255.0, 1 - blue, 0.0, blue)))
        b = int(round(mix(255.0, blue, 0.0, 1 - blue)))

    assert b + g + r == 255

    return b, g, r
