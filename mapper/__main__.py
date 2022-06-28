import cv2 as cv
import numpy as np

import mapper.util.kittidata as kd

import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.keypoint as kp
import mapper.vision.tracking as trck


def show_kitti_demo():
    kp.configure_keypoint(kp.KeypointType.AGAST)

    cv.namedWindow('viz')

    frame_nr = 0
    images = list()
    intrinsic_matrices = list()
    gt_poses = list()
    poses = list()

    test_dir = 'C:/Users/patri/repos/VisualSLAM/KITTI_sequence_2'
    for gray, proj_matrix, gt_pose in kd.KittiData(test_dir):
        instrinsic_matrix, _ = mat.decomp_pose_matrix(proj_matrix)
        points = kp.detect(gray, 1500)

        viz = im.visualization_image(gray)
        im.draw_features(viz, cv.KeyPoint_convert(points))

        cv.setWindowTitle('viz', f'Frame #{frame_nr}')
        cv.imshow('viz', viz)

        if len(images) > 0:
            ()
        else:
            # Start at zero.
            poses.append(mat.pose_matrix(
                np.eye(3, 3, dtype=np.float64), np.array([0.0, 0.0, 0.0])))

        frame_nr += 1
        images.append(gray)
        intrinsic_matrices.append(instrinsic_matrix)
        gt_poses.append(gt_pose)

        key = cv.waitKey(33)
        if key == 27:
            break

    cv.destroyAllWindows()


# def show_keypoints():
#     # kp.configure_keypoint(kp.KeypointType.AGAST)
#     kp.configure_keypoint(kp.KeypointType.AKAZE)
#     # kp.configure_keypoint(kp.KeypointType.ORB)

#     gray0 = im.scale_image(im.gray_convert(
#         im.read_image_bgr('C:/Users/patri/bilder/yaw0.jpeg')), 0.5)
#     gray1 = im.scale_image(im.gray_convert(
#         im.read_image_bgr('C:/Users/patri/bilder/yaw1.jpeg')), 0.5)

#     # Hack for iPhone images.
#     intrinsic_matrix = mat.intrinsic_matrix_35mm_film(26, im.image_size(gray0))

#     points0 = kp.SSC_refine(kp.detect(gray0), 5000, im.image_size(gray0))
#     points1 = kp.SSC_refine(kp.detect(gray1), 5000, im.image_size(gray1))

#     train = kp.compute(gray0, points0)
#     query = kp.compute(gray1, points1)

#     first_match = kp.match(train, query)
#     print(f"First match={len(first_match['train_keypoints'])}")

#     E, E_match = kp.E_refine(first_match, intrinsic_matrix)
#     print(f"E match={len(E_match['train_keypoints'])}")

#     xyz, ypr = trck.relative_pose(cv.KeyPoint_convert(E_match['train_keypoints']),
#                                   cv.KeyPoint_convert(
#                                       E_match['query_keypoints']),
#                                   E, intrinsic_matrix)

#     print(f'Query relative to train. xyz={xyz}, ypr={ypr}')

#     keys0 = im.visualization_image(gray0)
#     keys1 = im.visualization_image(gray1)
#     viz0 = im.visualization_image(gray0)
#     viz1 = im.visualization_image(gray1)

#     im.draw_features(keys0, np.array(cv.KeyPoint_convert(points0)))
#     im.draw_features(keys1, np.array(cv.KeyPoint_convert(points1)))
#     match = im.draw_matching_features(viz0, cv.KeyPoint_convert(E_match['train_keypoints']),
#                                       viz1, cv.KeyPoint_convert(E_match['query_keypoints']))

#     cv.namedWindow('Key points 0', cv.WINDOW_NORMAL +
#                    cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
#     cv.resizeWindow('Key points 0', im.image_size(keys0))
#     cv.imshow('Key points 0', keys0)

#     cv.namedWindow('Key points 1', cv.WINDOW_NORMAL +
#                    cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
#     cv.resizeWindow('Key points 1', im.image_size(keys1))
#     cv.imshow('Key points 1', keys1)

#     cv.namedWindow('Matching points', cv.WINDOW_NORMAL +
#                    cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
#     cv.resizeWindow('Matching points', im.image_size(match))
#     cv.imshow('Matching points', match)

#     cv.waitKey(0)
#     cv.destroyAllWindows()


def main():
    show_kitti_demo()
    # show_keypoints()


if __name__ == '__main__':
    main()
