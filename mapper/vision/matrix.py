import cv2 as cv
import numpy as np

import math

import mapper.vision.utils as utils


def ideal_intrinsic_matrix(fov: tuple, image_size: tuple) -> cv.Mat:
    """
    Compute an ideal intrinsic matrix (center at image center etc.).

    Parameters:
        fov: Tuple (horizontal fov, vertical fov) in degrees.
        image_size: Tuple (width, height) for the image (assuming pixels).

    Returns:
        A 3x3 intrinsic matrix.
    """
    assert isinstance(fov, tuple)
    assert len(fov) == 2
    assert isinstance(image_size, tuple)
    assert len(image_size) == 2

    h_fov, v_fov = fov
    w, h = image_size

    f_x = utils.focal_length_from_fov(h_fov, w)
    f_y = utils.focal_length_from_fov(v_fov, h)

    c_x = (w - 1) / 2.0
    c_y = (h - 1) / 2.0

    m = [f_x, 0.0, c_x, 0.0, f_y, c_y, 0.0, 0.0, 1.0]

    return np.array(m).reshape(3, 3)


def decomp_intrinsic_matrix(mat: np.ndarray) -> tuple:
    """
    Decompose the intrinsic matrix.

    Parameters:
        mat: A 3x3 intrinsic matrix.

    Returns:
        A nested tuple ((h_fov, vfov), (w, h)).
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 3)

    w = mat[0, 2] * 2.0 + 1
    h = mat[1, 2] * 2.0 + 1

    h_fov = utils.fov_from_focal_length(mat[0, 0], w)
    v_fov = utils.fov_from_focal_length(mat[1, 1], h)

    return ((h_fov, v_fov), (w, h))


def projection_matrix(intrinsic_mat: np.ndarray,
                      extrinsic_mat: np.ndarray) -> np.ndarray:
    """
    Create a projection matrix from instrinsic and extrinsic matrices.

    Parameters:
        intrinsic_mat: A 3x3 intrinsic matrix.
        extrinsic_mat: A 3x4 extrinsic matrix.

    Retrurns:
        A 3x4 projection matrix.
    """
    assert isinstance(intrinsic_mat, np.ndarray)
    assert intrinsic_mat.shape == (3, 3)
    assert isinstance(extrinsic_mat, np.ndarray)
    assert extrinsic_mat.shape == (3, 4)

    return intrinsic_mat @ extrinsic_mat


def pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create a pose matrix, holding the rotation and translation for something.

    Parameters:
        R: The rotation.
        t: The translation.

    Returns:
        A 3x4 pose matrix.
    """
    assert isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    assert isinstance(t, np.ndarray)
    assert len(t) == 3

    return np.hstack((R, t.reshape(3, 1)))


def decomp_pose_matrix(mat: np.ndarray) -> tuple:
    """
    Decompose a pose matrix into its R and t components.

    Parameters:
        mat: A 3x4 pose matrix.

    Returns:
        A tuple (R, t)
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 4)

    return (mat[:, :3].copy(), mat[:, 3].copy())


def decomp_pose_matrix_yprt_yxz(mat: np.ndarray) -> tuple:
    """
    Decompose a pose matrix into its ypr and t components.

    Parameters:
        mat: A 3x4 pose matrix.

    Returns:
        A tuple ((y, p, r), t).
    """
    R, t = decomp_pose_matrix(mat)

    return (decomp_ypr_matrix_yxz(R), t)


def homogeneous_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Create a homogeneous 4x4 matrix from a 3x4 matrix.

    Parameters:
        mat: A 3x4 matrix.

    Returns:
        A 4x4 homogeneous matrix.
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 4)

    return np.vstack((mat, np.array([0.0, 0.0, 0.0, 1.0])))


def decomp_homogeneous_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Decompose a homogeneous 4x4 matrix into a 3x4 matrix.

    Parameters:
        mat: 4x4 matrix.

    Returns:
        A 3x4 matrix.
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (4, 4)

    return mat[:3, :4].copy()


def fundamental_matrix(E: np.ndarray, intrinsic_mat: np.ndarray) -> np.ndarray:
    """
    Create a fundamental matrix from essential matrix and intrinsic matrix.

    Parameters:
        E: Essential matrix.
        intrinsic_mat: Intrinsic matrix.

    Returns:
        Fundamental matrix.
    """
    assert isinstance(E, np.ndarray)
    assert E.shape == (3, 3)
    assert isinstance(intrinsic_mat, np.ndarray)
    assert intrinsic_mat.shape == (3, 3)

    Kinv = np.linalg.inv(intrinsic_mat)

    return Kinv.T @ E @ Kinv


def intrinsic_matrix_35mm_film(focal_length: float, image_size: tuple) -> cv.Mat:
    """
    Helper function to compute an intrinsic matrix for a 35mm file (e.g. iphone Exif).
    """
    assert isinstance(image_size, tuple)
    assert len(image_size) == 2

    aspect_ratio = utils.aspect_ratio(image_size)
    h_fov = utils.fov_from_focal_length(focal_length, 35)
    v_fov = utils.fov_from_focal_length(focal_length, 35 / aspect_ratio)

    return ideal_intrinsic_matrix((h_fov, v_fov), image_size)


def extrinsic_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create an extrinsic matrix, transforming from world space
    to camera space.

    Parameters:
        R: 3x3 rotation matrix - describing the cameras rotation.
        t: camera position.

    Returns:
        3x4 matrix transforming to camera space.
    """
    assert isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    assert isinstance(t, np.ndarray)
    assert len(t) == 3

    return np.hstack((R.T, R.T @ -t.reshape(3, 1)))


def decomp_extrinsic_matrix(mat: np.ndarray) -> tuple:
    """
    Decompose an extrinsic matrix into R, t. Return new
    copies of data (i.e. not sharing data with extrinsic matrix).

    Parameters:
        mat: 3x4 extrinsic matrix.

    Returns:
        Tuple (R, t), similar to arguments for extrinsic data.
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 4)

    R = mat[:, :3].copy()
    t = mat[:, 3].copy()

    return (R.T, R.T @ -t)


def extrinsic_rtvec(R: np.ndarray, t: np.ndarray) -> tuple:
    """
    Create rvec, tvec as extrinsic data, transforming from world space
    to camera space.

    Parameters:
        R: 3x3 rotation matrix - describing the cameras rotation.
        t: camera position.

    Returns:
        Tuple (rvec, tvec).
    """
    rvec, jac = cv.Rodrigues(R.T)
    tvec = R.T @ -t

    return (rvec, tvec)


def decomp_extrinsic_rtvec(rvec: np.ndarray, tvec: np.ndarray) -> tuple:
    """
    Decompose extrinsic rvec, tvec into R, t.

    Parameters:
        rvec: Rotation vector.
        tvec: Translation vector.        

    Returns:
        Tuple (R, t), similar to arguments for extrinsic data.
    """
    assert isinstance(rvec, np.ndarray)
    assert len(rvec) == 3
    assert isinstance(tvec, np.ndarray)
    assert len(tvec) == 3

    R, jac = cv.Rodrigues(rvec)

    return (R.T, R.T @ -tvec)


def look_at_yxz(eye: np.ndarray, at: np.ndarray, down: np.ndarray) -> tuple:
    """
    Create a rotation matrix and t vector from look at vectorts. Suitable
    for OpenCV camera frames.
    """
    assert isinstance(eye, np.ndarray)
    assert len(eye) == 3
    assert isinstance(at, np.ndarray)
    assert len(at) == 3
    assert isinstance(down, np.ndarray)
    assert len(down) == 3

    z = at - eye
    z /= np.linalg.norm(z)

    down /= np.linalg.norm(down)

    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    return (np.hstack((x.reshape(3, 1), y.reshape(3, 1), z.reshape(3, 1))), eye)


def ypr_matrix_yxz(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Compute an Euler rotation matrix in order y, x and z. Suitable for OpenCV
    camera frames.

    Parameters:
        yaw: Yaw angle in degrees.
        pitch: Pitch angle in degrees.
        roll: Roll angle in degrees.

    Returns:
        A 3x3 rotation matrix.
    """
    y = np.radians(yaw)
    x = np.radians(pitch)
    z = np.radians(roll)

    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    cz = math.cos(z)
    sz = math.sin(z)

    mat = [cy * cz + sx * sy * sz, cz * sx * sy - cy * sz, cx * sy,
           cx * sz, cx * cz, -sx,
           -cz * sy + cy * sx * sz, cy * cz * sx + sy * sz, cx * cy
           ]

    return np.array(mat).reshape(3, 3)


def decomp_ypr_matrix_yxz(mat: np.ndarray) -> tuple:
    """
    Decompose an Euler rotation matrix in order y, x, and z into
    yaw, pitch and roll.

    Parameters:
        Tuple (yaw, pitch, roll) in degrees.
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 3)

    y = math.atan2(mat[0, 2], mat[2, 2])
    x = math.asin(-mat[1, 2])
    z = math.atan2(mat[1, 0], mat[1, 1])

    return (math.degrees(y), math.degrees(x), math.degrees(z))


def ypr_matrix_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Compute an Euler rotation matrix in order z, y and z. Suitable for ECEF
    camera frames.

    Parameters:
        yaw: Yaw angle in degrees.
        pitch: Pitch angle in degrees.
        roll: Roll angle in degrees.

    Returns:
        A 3x3 rotation matrix.
    """
    z = math.radians(yaw)
    y = math.radians(pitch)
    x = math.radians(roll)

    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)

    mat = [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz,
           cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz,
           -sy, cy * sx, cx * cy
           ]

    return np.array(mat).reshape(3, 3)


def decomp_ypr_matrix_zyx(mat: np.ndarray) -> tuple:
    """
    Decompose an Euler rotation matrix in order z, y, and x into
    yaw, pitch and roll.

    Parameters:
        Tuple (yaw, pitch, roll) in degrees.
    """
    z = math.atan2(mat[1, 0], mat[0, 0])
    y = math.asin(-mat[2, 0])
    x = math.atan2(mat[2, 1], mat[2, 2])

    return (math.degrees(z), math.degrees(y), math.degrees(x))


def ecef_to_camera_matrix() -> np.ndarray:
    """
    Compute a matrix that maps from ECEF frame to
    a OpenCV camera frame (withing the ECEF frame).

    Returns:
        3x3 rotation matrix.
    """
    mat = [0.0, 1.0, 0.0,
           0.0, 0.0, -1.0,
           -1.0, 0.0, 0.0]

    return np.array(mat).reshape(3, 3)
