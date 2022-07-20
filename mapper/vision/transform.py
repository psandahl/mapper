import cv2 as cv
import numpy as np

import random

import mapper.vision.matrix as mat


def remap_frame(R: np.ndarray, map: np.ndarray) -> np.ndarray:
    """
    Remap a rotation frame, e.g. ECEF, to be a camera frame within
    ECEF. If no mapping shall be made, e.g. camera and world frames
    have the same axes, map can be set to identity.

    Parameters:
        R: 3x3 rotation matrix.
        map: 3x3 rotation matrix.
    """
    assert isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    assert isinstance(map, np.ndarray)
    assert map.shape == (3, 3)

    return R @ map.T


def world_to_camera_mat(ext: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a world coordinate to a camera coordinate using an extrinsic matrix.

    Parameters:
        ext: The extrinsic matrix.
        xyz: The world coordinate.

    Returns:
        The camera coordinate.
    """
    assert isinstance(ext, np.ndarray)
    assert ext.shape == (3, 4)
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    xyz_h = np.append(xyz, 1.0)
    return ext @ xyz_h


def infront_of_camera(ext: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    _, _, z = world_to_camera_mat(ext, xyz)
    return z > 0.0


def camera_to_world_mat(ext: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a camera coordinate to a world coordinate using an extrinsic matrix.

    Parameters:
        ext: The extrinsic matrix.
        xyz: The camera coordinate.

    Returns:
        The world coordinate.
    """
    assert isinstance(ext, np.ndarray)
    assert ext.shape == (3, 4)
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    # Decomp gives camera to world from start.
    R, t = mat.decomp_extrinsic_matrix(ext)

    return R @ xyz + t


def world_to_camera_rtvec(rvec: np.ndarray, tvec: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a world coordinate to a camera coordinate using rvec and tvec.

    Parameters:
        rvec: Rotation vector.
        tvec: Translation vector.        
        xyz: The world coordinate.

    Returns:
        The camera coordinate.
    """
    assert isinstance(rvec, np.ndarray)
    assert len(rvec) == 3
    assert isinstance(tvec, np.ndarray)
    assert len(tvec) == 3
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    R, _ = cv.Rodrigues(rvec)

    return R @ xyz + tvec


def camera_to_world_rtvec(rvec: np.ndarray, tvec: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Transform a camera coordinate to a world coordinate using rvec and tvec.

    Parameters:
        rvec: Rotation vector.
        tvec: Translation vector.        
        xyz: The camera coordinate.

    Returns:
        The world coordinate.
    """
    assert isinstance(rvec, np.ndarray)
    assert len(xyz) == 3
    assert isinstance(tvec, np.ndarray)
    assert len(tvec) == 3
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    R, _ = cv.Rodrigues(rvec)

    # rtvec always go from world to camera - so invert!
    return R.T @ xyz + R.T @ -tvec


def change_pose(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Change the pose/base by applying the delta to the pose.

    Parameters:
        pose: The pose to change.
        delta: The pose to apply.

    Returns:
        The new pose.
    """
    assert isinstance(pose, np.ndarray)
    assert pose.shape == (3, 4)
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (3, 4)

    new_pose = mat.homogeneous_matrix(
        pose) @ np.linalg.inv(mat.homogeneous_matrix(delta))

    return mat.decomp_homogeneous_matrix(new_pose)


def invert_3x4_matrix(m: np.ndarray) -> np.ndarray:
    """
    Invert an 3x4 matrix (e.g. pose or extrinsic).

    Parameters:
        m: The matrix.

    Returns:
        The inverted matrix.
    """
    assert isinstance(m, np.ndarray)
    assert m.shape == (3, 4)

    return mat.decomp_homogeneous_matrix(np.linalg.inv(mat.homogeneous_matrix(m)))


def project_point(projection_mat: np.ndarray, xyz: np.ndarray) -> tuple:
    """
    Project a world coordinate to image.

    Parameters:
        projection_mat: A 3x4 projection matrix.
        xyz: The world point.

    Returns:
        A tuple (px, z depth).
    """
    assert isinstance(projection_mat, np.ndarray)
    assert projection_mat.shape == (3, 4)
    assert isinstance(xyz, np.ndarray)
    assert len(xyz) == 3

    xyz_h = np.append(xyz, 1.0)
    px = projection_mat @ xyz_h
    z = px[2]
    px /= z

    return px[:2], z


def unproject_point(inv_intrinsic: np.ndarray, inv_extrinsic: np.ndarray,
                    px: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Unproject an image point to world coordinate, using depth value.

    Parameters:
        inv_intrinsic: Inverted intrinsic matrix.
        inv_extrinsic: Inverted extrinsic matrix.
        px: Image coordinate.
        z: Depth value.

    Returns:
        World point.
    """
    assert isinstance(inv_intrinsic, np.ndarray)
    assert inv_intrinsic.shape == (3, 3)
    assert isinstance(inv_extrinsic, np.ndarray)
    assert inv_extrinsic.shape == (3, 4)
    assert len(px) == 2

    px_h = np.append(px, 1.0)
    xyz = inv_intrinsic @ px_h

    return inv_extrinsic @ np.append(xyz * z, 1.0)


def select_points_for_pose(image_points: np.ndarray, world_points: np.ndarray,
                           intrinsic_mat: np.ndarray) -> tuple:
    assert isinstance(image_points, list)
    assert isinstance(world_points, list)
    assert len(image_points) > 3
    assert len(image_points) == len(world_points)
    assert isinstance(intrinsic_mat, np.ndarray)
    assert intrinsic_mat.shape == (3, 3)

    indices = list(range(0, len(image_points)))

    inliers = list()
    current = list()

    max_iter = 100
    error_threshold = 0.25
    for iter in range(0, max_iter):
        current.clear()

        # Select points.
        i_0, i_1, i_2 = random.sample(indices, 3)
        w_points = np.array(
            [world_points[i_0], world_points[i_1], world_points[i_2]])
        i_points = np.array(
            [image_points[i_0], image_points[i_1], image_points[i_2]])

        # Solve for P3P, and select the solution with least error.
        solutions, rvecs, tvecs = cv.solveP3P(
            w_points, i_points, intrinsic_mat, None, cv.SOLVEPNP_AP3P)

        if solutions == 0:
            # No solutions. Try next iteration.
            continue

        rvec, tvec = rvecs[0], tvecs[0]

        # Create a projection matrix.
        R, _ = cv.Rodrigues(rvec)
        extrinsic_mat = np.hstack((R, tvec))
        projection_mat = mat.projection_matrix(intrinsic_mat, extrinsic_mat)

        # Project all points to identify inliers.
        for index, xyz in enumerate(world_points):
            px, _ = project_point(projection_mat, xyz)
            error = np.linalg.norm(px - image_points[index])
            if error < error_threshold:
                current.append(index)

        if len(current) > len(inliers):
            # Swap lists if current is larger.
            inliers, current = current, inliers

        if len(inliers) == len(image_points):
            # Cannot be more inliers.
            break

    print(
        f'select points for pose (RANSAC): from {len(image_points)} pairs, {len(inliers)} are selected')

    selected_image_points = list()
    selected_world_points = list()
    for inlier in inliers:
        selected_image_points.append(image_points[inlier])
        selected_world_points.append(world_points[inlier])

    return selected_image_points, selected_world_points


def project_points_opt_6dof(image_points: np.ndarray, world_points: np.ndarray,
                            intrinsic_mat: np.ndarray, pose: np.ndarray,
                            delta: np.ndarray) -> list:
    """
    Projection function to be used for optimization of 6DoF camera pose.
    """
    assert isinstance(image_points, list)
    assert isinstance(world_points, list)
    assert len(image_points) == len(world_points)
    assert isinstance(intrinsic_mat, np.ndarray)
    assert intrinsic_mat.shape == (3, 3)
    assert isinstance(pose, np.ndarray)
    assert pose.shape == (3, 4)
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (6,)

    R, _ = cv.Rodrigues(delta[:3])
    t = delta[3:]

    delta_pose = mat.pose_matrix(R, t)
    full_pose = change_pose(pose, delta_pose)

    R, t = mat.decomp_pose_matrix(full_pose)
    extrinsic_mat = mat.extrinsic_matrix(R, t)
    projection_mat = mat.projection_matrix(intrinsic_mat, extrinsic_mat)

    err = list()
    for index, xyz in enumerate(world_points):
        px, _ = project_point(projection_mat, xyz)
        err.append(np.linalg.norm(px - image_points[index]))

    # print(
    #    f'project_points_opt_6dof. sum(err)={np.sum(err)}, mean(err)={np.mean(err)}, max(err)={np.max(err)}')

    return err
