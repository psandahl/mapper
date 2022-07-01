import cv2 as cv
import numpy as np

import mapper.util.kittidata as kd
import mapper.util.misc as misc
from mapper.util.panel import Panel

import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.keypoint as kp
import mapper.vision.tracking as trk
import mapper.vision.transform as trf


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

    panel = Panel()

    images = list()
    intrinsic_matrices = list()
    descriptor_pairs = list()
    gt_poses = list()
    poses = list()

    # Iterate through dataset.
    for image, proj_matrix, gt_pose in kd.KittiData(test_dir):
        frame_nr = len(images)

        # Unpack the intrinsic matrix.
        instrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)

        # Compute keypoints and descriptors.
        frame_keypoints = kp.detect(image, 650)
        frame_descriptor_pair = kp.compute(image, frame_keypoints)

        if frame_nr > 0:
            # Do stuff.
            prev_image = im.visualization_image(misc.last_in(images))
            prev_pose = misc.last_in(poses)
            prev_descriptor_pair = misc.last_in(descriptor_pairs)

            match = kp.match(prev_descriptor_pair, frame_descriptor_pair)
            rel_pose, pose_match = trk.visual_pose_prediction(
                match, instrinsic_matrix)

            pose = trf.change_pose(prev_pose, rel_pose)
            poses.append(pose)

            print_pose_comparision(f'frame={frame_nr}',
                                   mat.decomp_pose_matrix_yprt_yxz(pose),
                                   mat.decomp_pose_matrix_yprt_yxz(gt_pose))

            panel.set_caption(f'frame={frame_nr}')
            panel.set_pose_matches(prev_image,
                                   cv.KeyPoint_convert(
                                       prev_descriptor_pair[0]),
                                   cv.KeyPoint_convert(
                                       pose_match['train_keypoints']),
                                   image,
                                   cv.KeyPoint_convert(
                                       frame_descriptor_pair[0]),
                                   cv.KeyPoint_convert(pose_match['query_keypoints']))
            panel.update()

            key = cv.waitKey(0)
            if key == 27:
                break
        else:
            # This is the first frame. Use the ground truth pose.
            poses.append(gt_pose)

        # Save stuff.
        images.append(image)
        intrinsic_matrices.append(instrinsic_matrix)
        descriptor_pairs.append(frame_descriptor_pair)
        gt_poses.append(gt_pose)

    panel.destroy_window()


def main():
    # tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_1')
    # tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_2')
    tracking('C:\\Users\\patri\\kitti\\KITTI_sequence_long_1')


if __name__ == '__main__':
    main()
