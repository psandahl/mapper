import cv2 as cv
import numpy as np

import mapper.util.kittidata as kd

import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.keypoint as kp
import mapper.vision.tracking as trck


def last_in(xs: list) -> any:
    return xs[len(xs) - 1]


def tracking(test_dir: str) -> None:
    kp.configure_keypoint(kp.KeypointType.AKAZE)

    cv.namedWindow('viz')
    cv.namedWindow('match')

    frame_nr = 0
    images = list()
    intrinsic_matrices = list()
    kpt_pairs = list()
    gt_poses = list()
    poses = list()

    for gray, proj_matrix, gt_pose in kd.KittiData(test_dir):
        instrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)
        points = kp.detect(gray, 650)
        query = kp.compute(gray, points)

        viz = im.visualization_image(gray)
        im.draw_features(viz, cv.KeyPoint_convert(points))

        cv.setWindowTitle('viz', f'Frame #{frame_nr}')
        cv.imshow('viz', viz)

        if len(images) > 0:
            # Do stuff.
            prev_image = im.visualization_image(last_in(images))
            prev_pose = last_in(poses)
            train = last_in(kpt_pairs)

            match = kp.match(train, query)
            rel_pose, pose_match = trck.visual_pose_prediction(
                match, instrinsic_matrix)

            prev_pose_hom = mat.homogeneous_matrix(prev_pose)
            rel_pose_hom = mat.homogeneous_matrix(rel_pose)

            pose_hom = prev_pose_hom @ np.linalg.inv(rel_pose_hom)
            pose = mat.decomp_homogeneous_matrix(pose_hom)

            poses.append(pose)

            R, t = mat.decomp_pose_matrix(pose)
            R_gt, t_gt = mat.decomp_pose_matrix(gt_pose)
            ypr = mat.decomp_ypr_matrix_yxz(R)
            ypr_gt = mat.decomp_ypr_matrix_yxz(R_gt)

            print(f'Position={t}, GT position={t_gt}')
            print(f'YPR={ypr}, GT YPR={ypr_gt}')

            match_img = im.draw_matching_features(prev_image,
                                                  cv.KeyPoint_convert(
                                                      pose_match['train_keypoints']),
                                                  viz,
                                                  cv.KeyPoint_convert(pose_match['query_keypoints']))

            cv.imshow('match', match_img)
        else:
            # Start at zero.
            poses.append(mat.pose_matrix(
                np.eye(3, 3, dtype=np.float64), np.array([0.0, 0.0, 0.0])))

        frame_nr += 1
        images.append(gray)
        intrinsic_matrices.append(instrinsic_matrix)
        kpt_pairs.append(query)
        gt_poses.append(gt_pose)

        key = cv.waitKey(0)
        if key == 27:
            break

    cv.destroyAllWindows()


def main():
    tracking('C:/Users/patri/repos/VisualSLAM/KITTI_sequence_2')


if __name__ == '__main__':
    main()
