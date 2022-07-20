import cv2 as cv
import numpy as np

import os
import sys

import mapper.util.kittidata as kd
import mapper.util.misc as misc
from mapper.util.panel import Panel

import mapper.vision.epipolar as epi
from mapper.vision.frame import Frame
import mapper.vision.flow as flow
import mapper.vision.matrix as mat
import mapper.vision.keypoint as kp
import mapper.vision.mapping as map
import mapper.vision.tracking as trk
import mapper.vision.transform as trf


def print_pose_comparision(label: str, pose: np.ndarray, gt_pose: np.ndarray) -> None:
    ypr, t = mat.decomp_pose_matrix_yprt_yxz(pose)
    ypr_gt, t_gt = mat.decomp_pose_matrix_yprt_yxz(gt_pose)
    dist = np.linalg.norm(t - t_gt)

    print(f'Pose comparision - {label}')
    print(' Position:')
    print(f'  VO x={t[0]:+.2f} y={t[1]:+.2f} z={t[2]:+.2f}')
    print(f'  GT x={t_gt[0]:+.2f} y={t_gt[1]:+.2f} z={t_gt[2]:+.2f}')
    print(f'  Euclidean distance={dist:.2f}m')
    print(' Rotation:')
    print(f'  VO y={ypr[0]:+.2f} p={ypr[1]:+.2f} r={ypr[2]:+.2f}')
    print(f'  GT y={ypr_gt[0]:+.2f} p={ypr_gt[1]:+.2f} r={ypr_gt[2]:+.2f}')


def pose_distance(pose_0: np.ndarray, pose_1: np.ndarray) -> float:
    _, t_0 = mat.decomp_pose_matrix(pose_0)
    _, t_1 = mat.decomp_pose_matrix(pose_1)

    return np.linalg.norm(t_1 - t_0)


def test_epipolar_search(F, train_image, train_points, query_image, query_points):
    distances = list()
    for index, train_point in enumerate(train_points):
        line = epi.epipolar_line(F, train_point)
        result = epi.search_line(
            line, train_point, train_image, query_image)
        if not result is None:
            found, error = result
            distance = np.linalg.norm(found - query_points[index])
            if distance > 10:
                print(
                    f'SAD={error} found={found} should be={query_points[index]}')

            distances.append(distance)

    print(
        f'Max dist={np.max(distances)} min={np.min(distances)} mean={np.mean(distances)}')


def plk_tracking_and_mapping(data_dir: str) -> None:
    data_extent = misc.read_2d_box_from_3x4_matrices(
        os.path.join(data_dir, 'poses.txt'))

    panel = Panel(data_extent)

    images = list()
    intrinsic_matrices = list()
    gt_poses = list()
    est_poses = list()

    # Iterate through dataset.
    for image, proj_matrix, gt_pose in kd.KittiData(data_dir):
        frame_id = len(images)

        print(f'Processing frame={frame_id}')

        # Unpack the intrinsic matrix.
        intrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)

        if frame_id > 0:
            # Do stuff.
            prev_image = images[-1]
            prev_pose = est_poses[-1]

            train_points = flow.points_to_track(prev_image)
            train_match, query_match = flow.sparse_optical_flow(
                prev_image, image, train_points)

            print(
                f"Number of matching points={len(train_match)}")
            rel_pose, pose_train_match, pose_query_match, E = trk.visual_pose_prediction_plk(
                train_match, query_match, intrinsic_matrix)
            print(
                f"Number of pose prediction inliers={len(pose_train_match)}")

            pose = trf.change_pose(prev_pose, rel_pose)
            est_poses.append(pose)

            print_pose_comparision('prediction', pose, gt_pose)

            F = mat.fundamental_matrix(E, intrinsic_matrix)
            test_epipolar_search(
                F, prev_image, pose_train_match, image, pose_query_match)

            panel.set_caption(f'frame={frame_id}')
            panel.set_pose_matches(prev_image,
                                   train_match,
                                   np.array(pose_train_match),
                                   image,
                                   query_match,
                                   np.array(pose_query_match),
                                   F
                                   )

            panel.add_camera(gt_pose)
            panel.add_camera(pose, color=(255, 0, 0))
            panel.update()

            key = cv.waitKey(0)
            if key == 27:
                break
        else:
            # This is the first frame. Use the ground truth pose.
            est_poses.append(gt_pose)

        # Save stuff.
        images.append(image)
        intrinsic_matrices.append(intrinsic_matrix)
        gt_poses.append(gt_pose)

    print('Track is complete. Press ENTER to quit')
    sys.stdin.read(1)

    panel.destroy_window()


def plk_tracking(data_dir: str) -> None:
    data_extent = misc.read_2d_box_from_3x4_matrices(
        os.path.join(data_dir, 'poses.txt'))

    panel = Panel(data_extent)

    images = list()
    intrinsic_matrices = list()
    gt_poses = list()
    est_poses = list()

    # Iterate through dataset.
    for image, proj_matrix, gt_pose in kd.KittiData(data_dir):
        frame_id = len(images)

        print(f'Processing frame={frame_id}')

        # Unpack the intrinsic matrix.
        intrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)

        if frame_id > 0:
            # Do stuff.
            prev_image = images[-1]
            prev_pose = est_poses[-1]

            train_points = flow.points_to_track(prev_image)
            train_match, query_match = flow.sparse_optical_flow(
                prev_image, image, train_points)

            print(
                f"Number of matching points={len(train_match)}")
            rel_pose, pose_train_match, pose_query_match, E = trk.visual_pose_prediction_plk(
                train_match, query_match, intrinsic_matrix)
            print(
                f"Number of pose prediction inliers={len(pose_train_match)}")

            pose = trf.change_pose(prev_pose, rel_pose)
            est_poses.append(pose)

            print_pose_comparision('prediction', pose, gt_pose)

            F = mat.fundamental_matrix(E, intrinsic_matrix)

            panel.set_caption(f'frame={frame_id}')
            panel.set_pose_matches(prev_image,
                                   train_match,
                                   np.array(pose_train_match),
                                   image,
                                   query_match,
                                   np.array(pose_query_match),
                                   F
                                   )

            panel.add_camera(gt_pose)
            panel.add_camera(pose, color=(255, 0, 0))
            panel.update()

            key = cv.waitKey(15)
            if key == 27:
                break
        else:
            # This is the first frame. Use the ground truth pose.
            est_poses.append(gt_pose)

        # Save stuff.
        images.append(image)
        intrinsic_matrices.append(intrinsic_matrix)
        gt_poses.append(gt_pose)

    print('Track is complete. Press ENTER to quit')
    sys.stdin.read(1)

    panel.destroy_window()


def feature_tracking(data_dir: str) -> None:
    kp.configure_keypoint(kp.KeypointType.AKAZE)

    data_extent = misc.read_2d_box_from_3x4_matrices(
        os.path.join(data_dir, 'poses.txt'))

    panel = Panel(data_extent)

    images = list()
    intrinsic_matrices = list()
    descriptor_pairs = list()
    gt_poses = list()
    est_poses = list()

    # Iterate through dataset.
    for image, proj_matrix, gt_pose in kd.KittiData(data_dir):
        frame_id = len(images)

        print(f'Processing frame={frame_id}')

        # Unpack the intrinsic matrix.
        instrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)

        # Compute keypoints and descriptors.
        frame_keypoints = kp.detect(image, 650)
        frame_descriptor_pair = kp.compute(image, frame_keypoints)

        if frame_id > 0:
            # Do stuff.
            prev_image = images[-1]
            prev_pose = est_poses[-1]
            prev_descriptor_pair = descriptor_pairs[-1]

            match = kp.match(prev_descriptor_pair, frame_id - 1,
                             frame_descriptor_pair, frame_id)
            print(
                f"Number of matching keypoints={len(match['query_keypoints'])}")
            rel_pose, pose_match = trk.visual_pose_prediction_kpt(match,
                                                                  instrinsic_matrix)
            print(
                f"Number of pose prediction inliers={len(pose_match['query_keypoints'])}")

            pose = trf.change_pose(prev_pose, rel_pose)
            est_poses.append(pose)

            print_pose_comparision('prediction', pose, gt_pose)

            panel.set_caption(f'frame={frame_id}')
            panel.set_pose_matches(prev_image,
                                   cv.KeyPoint_convert(
                                       prev_descriptor_pair[0]),
                                   cv.KeyPoint_convert(
                                       pose_match['train_keypoints']),
                                   image,
                                   cv.KeyPoint_convert(
                                       frame_descriptor_pair[0]),
                                   cv.KeyPoint_convert(pose_match['query_keypoints']))

            panel.add_camera(gt_pose)
            panel.add_camera(pose, color=(255, 0, 0))
            panel.update()

            key = cv.waitKey(15)
            if key == 27:
                break
        else:
            # This is the first frame. Use the ground truth pose.
            est_poses.append(gt_pose)

        # Save stuff.
        images.append(image)
        intrinsic_matrices.append(instrinsic_matrix)
        descriptor_pairs.append(frame_descriptor_pair)
        gt_poses.append(gt_pose)

    print('Track is complete. Press ENTER to quit')
    sys.stdin.read(1)

    panel.destroy_window()


def tracking_and_mapping(data_dir: str, cheat_frames: int = 5) -> None:
    kp.configure_keypoint(kp.KeypointType.AKAZE)

    data_extent = misc.read_2d_box_from_3x4_matrices(
        os.path.join(data_dir, 'poses.txt'))

    panel = Panel(data_extent)

    images = list()
    intrinsic_matrices = list()
    descriptor_pairs = list()
    pose_matches = list()
    gt_poses = list()
    est_poses = list()
    landmarks = list()

    # Iterate through dataset.
    for image, proj_matrix, gt_pose in kd.KittiData(data_dir):
        frame_id = len(images)

        new_landmarks = list()

        print(f'Processing frame={frame_id}')

        # Unpack the intrinsic matrix.
        instrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)

        # Compute keypoints and descriptors.
        frame_keypoints = kp.detect(image, 650)
        frame_descriptor_pair = kp.compute(image, frame_keypoints)

        if frame_id > 0:
            # Do stuff.
            prev_image = images[-1]
            prev_pose = est_poses[-1]
            prev_descriptor_pair = descriptor_pairs[-1]

            # Get a keypoint match with the previous frame.
            match = kp.match(prev_descriptor_pair, frame_id - 1,
                             frame_descriptor_pair, frame_id)
            print(
                f"Number of matching keypoints={len(match['query_keypoints'])}")

            scale = 1.0
            if frame_id <= cheat_frames:
                # To get the first triangulations right, bootstrap with
                # help of the ground truth.
                scale = pose_distance(gt_poses[-1], gt_pose)
                print(f'Cheating with scale from GT={scale}')

            # Using the match, predict the pose for this frame.
            rel_pose, pose_match = trk.visual_pose_prediction_kpt(match,
                                                                  instrinsic_matrix,
                                                                  scale=scale)

            print(
                f"Number of pose prediction inliers={len(pose_match['query_keypoints'])}")

            pose = trf.change_pose(prev_pose, rel_pose)

            print_pose_comparision('prediction', pose, gt_pose)

            # If there's at least two previous frames, we can map from the history
            # and do a better estimation for the current frame using the map.
            if len(images) > 1:
                # Note: This requires adjacent frames atm.
                new_landmarks = map.sparse_mapping(frame_id,
                                                   est_poses[-2], intrinsic_matrices[-2], images[-2],
                                                   est_poses[-1], intrinsic_matrices[-1], images[-1],
                                                   pose_matches[-1])
                map.add_new_landmarks(landmarks, new_landmarks)

                rel_pose = trk.landmark_pose_estimation(frame_id, landmarks,
                                                        frame_descriptor_pair,
                                                        instrinsic_matrix, pose, image)
                pose = trf.change_pose(pose, rel_pose)
                print_pose_comparision('estimation', pose, gt_pose)

            # Visualize stuff.
            panel.set_caption(f'frame={frame_id}')
            panel.set_pose_matches(prev_image,
                                   cv.KeyPoint_convert(
                                       prev_descriptor_pair[0]),
                                   cv.KeyPoint_convert(
                                       pose_match['train_keypoints']),
                                   image,
                                   cv.KeyPoint_convert(
                                       frame_descriptor_pair[0]),
                                   cv.KeyPoint_convert(pose_match['query_keypoints']))

            panel.add_camera(gt_pose)
            panel.add_camera(pose, color=(255, 0, 0))
            panel.add_landmarks(new_landmarks)
            panel.update()

            # Update lists with pose match with previous frame and the pose.
            pose_matches.append(pose_match)
            est_poses.append(pose)

            key = cv.waitKey(30)
            if key == 27:
                break
        else:
            # This is the first frame. Use the ground truth pose.
            pose_matches.append(None)
            est_poses.append(gt_pose)

        # Save stuff.
        images.append(image)
        intrinsic_matrices.append(instrinsic_matrix)
        descriptor_pairs.append(frame_descriptor_pair)
        gt_poses.append(gt_pose)

        # Manage landmarks.
        landmarks = map.manage_landmarks(frame_id, landmarks)

    print('Track is complete. ENTER to quit')
    sys.stdin.read(1)

    panel.destroy_window()


def run_mapper_from_kitti_data(data_dir: str) -> None:
    frame_id = 0
    keyframes = list()
    previous_frame = None

    # Iterate through dataset.
    for image, proj_matrix, gt_pose in kd.KittiData(data_dir):
        intrinsic_mat, _ = mat.decomp_pose_matrix(proj_matrix)
        keyframe = keyframes[-1] if len(keyframes) > 0 else None

        # Create a new frame from the input.
        frame = Frame(frame_id, image, intrinsic_mat)

        # Track the frame against the previous frame.
        if not previous_frame is None:
            frame.track_against(previous_frame)

        print_pose_comparision(f'#{frame_id}', frame.pose_mat, gt_pose)

        # Use the frame for mapping.
        if not keyframe is None:
            keyframe.map_with(frame)

        # Check if the current frame should be promoted ...
        if frame.should_be_promoted(keyframe):
            # Yes! Promote and store.
            frame.promote_to_keyframe(keyframe)
            keyframes.append(frame)

        # Delete the previous frame, if any.
        if not previous_frame is None:
            del previous_frame

        # Store the current frame until next iteration.
        previous_frame = frame

        # Increase id.
        frame_id += 1

    print(f'Done. Num keyframes={len(keyframes)}')


def main():
    run_mapper_from_kitti_data('C:\\Users\\patri\\kitti\\KITTI_sequence_2')

    # plk_tracking_and_mapping('C:\\Users\\patri\\kitti\\KITTI_sequence_2')
    # plk_tracking_and_mapping('C:\\Users\\patri\\kitti\\parking\\parking')

    # plk_tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_1')
    # plk_tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_2')
    # plk_tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_long_1')
    # plk_tracking('C:\\Users\\patri\\kitti\\parking\\parking')

    # feature_tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_1')
    # feature_tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_2')
    # feature_tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_long_1')
    # feature_tracking('C:\\Users\\patri\\kitti\\parking\\parking')

    # tracking_and_mapping('C:\\Users\\patri\\kitti\\KITTI_sequence_1')
    # tracking_and_mapping('C:\\Users\\patri\\kitti\\KITTI_sequence_2')
    # tracking_and_mapping('C:\\Users\\patri\\kitti\\KITTI_sequence_long_1')
    # tracking_and_mapping('C:\\Users\\patri\\kitti\\parking\\parking')


if __name__ == '__main__':
    main()
