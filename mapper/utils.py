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
