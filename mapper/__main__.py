import cv2 as cv
import numpy as np

import mapper.util.kittidata as kd

import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.keypoint as kp
import mapper.vision.tracking as trk
import mapper.vision.transform as trf


def last_in(xs: list) -> any:
    return xs[len(xs) - 1]


def print_pose_comparision(label: str, yprt: tuple, yprt_gt: tuple) -> None:
    ypr, t = yprt
    ypr_gt, t_gt = yprt_gt
    dist = np.linalg.norm(t - t_gt)

    print(f'Pose comparision {label}')
    print(' Position:')
    print(f'  VO x={t[0]:+.2f} y={t[1]:+.2f} z={t[2]:+.2f}')
    print(f'  GT x={t_gt[0]:+.2f} y={t_gt[1]:+.2f} z={t_gt[2]:+.2f}')
    print(f'  Euclidean distance={dist:.2f}m')
    print(' Rotation:')
    print(f'  VO y={ypr[0]:+.2f} p={ypr[1]:+.2f} r={ypr[2]:+.2f}')
    print(f'  GT y={ypr_gt[0]:+.2f} p={ypr_gt[1]:+.2f} r={ypr_gt[2]:+.2f}')


def tracking(test_dir: str) -> None:
    kp.configure_keypoint(kp.KeypointType.AKAZE)

    cv.namedWindow('viz')
    cv.namedWindow('match')
    images = list()
    intrinsic_matrices = list()
    descriptor_pairs = list()
    gt_poses = list()
    poses = list()

    # Iterate through dataset.
    for gray, proj_matrix, gt_pose in kd.KittiData(test_dir):
        frame_nr = len(images)

        # Unpack the intrinsic matrix.
        instrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)

        # Compute keypoints and descriptors.
        frame_keypoints = kp.detect(gray, 650)
        frame_descriptor_pair = kp.compute(gray, frame_keypoints)

        viz = im.visualization_image(gray)
        im.draw_features(viz, cv.KeyPoint_convert(frame_keypoints))

        cv.setWindowTitle('viz', f'Frame #{frame_nr}')
        cv.imshow('viz', viz)

        if frame_nr > 0:
            # Do stuff.
            prev_image = im.visualization_image(last_in(images))
            prev_pose = last_in(poses)
            prev_descriptor_pair = last_in(descriptor_pairs)

            match = kp.match(prev_descriptor_pair, frame_descriptor_pair)
            rel_pose, pose_match = trk.visual_pose_prediction(
                match, instrinsic_matrix)

            pose = trf.change_pose(prev_pose, rel_pose)
            poses.append(pose)

            match_img = im.draw_matching_features(prev_image,
                                                  cv.KeyPoint_convert(
                                                      pose_match['train_keypoints']),
                                                  viz,
                                                  cv.KeyPoint_convert(pose_match['query_keypoints']))

            cv.imshow('match', match_img)

            print_pose_comparision(f'frame={frame_nr}',
                                   mat.decomp_pose_matrix_yprt_yxz(pose),
                                   mat.decomp_pose_matrix_yprt_yxz(gt_pose))
        else:
            # This is the first frame. Use the ground truth pose.
            poses.append(gt_pose)

        # Save stuff.
        images.append(gray)
        intrinsic_matrices.append(instrinsic_matrix)
        descriptor_pairs.append(frame_descriptor_pair)
        gt_poses.append(gt_pose)

        key = cv.waitKey(0)
        if key == 27:
            break

    cv.destroyAllWindows()


def main():
    tracking('C:/Users/patri/repos/VisualSLAM/KITTI_sequence_1')


if __name__ == '__main__':
    main()
