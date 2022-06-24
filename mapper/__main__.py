import cv2 as cv
import numpy as np

import mapper.image as im
import mapper.matrix as mat
import mapper.keypoint as kp


def show_keypoints():
    gray0 = im.scale_image(im.gray_convert(
        im.read_image_bgr('c:/Users/patri/bilder/IMG_0141.jpeg')), 0.5)
    gray1 = im.scale_image(im.gray_convert(
        im.read_image_bgr('c:/Users/patri/bilder/IMG_0142.jpeg')), 0.5)

    intrinsic_matrix = mat.intrinsic_matrix_35mm_film(26, im.image_size(gray0))

    points0 = kp.SSC_refine(kp.detect(gray0), 5000, im.image_size(gray0))
    points1 = kp.SSC_refine(kp.detect(gray1), 5000, im.image_size(gray1))

    train = kp.compute(gray0, points0)
    query = kp.compute(gray1, points1)

    first_match = kp.match(train, query)
    print(f"First match={len(first_match['train_keypoints'])}")

    E, E_match = kp.E_refine(first_match, intrinsic_matrix, 0.1)
    print(f"E match={len(E_match['train_keypoints'])}")

    keys0 = im.visualization_image(gray0)
    keys1 = im.visualization_image(gray1)
    viz0 = im.visualization_image(gray0)
    viz1 = im.visualization_image(gray1)

    im.draw_features(keys0, np.array(cv.KeyPoint_convert(points0)))
    im.draw_features(keys1, np.array(cv.KeyPoint_convert(points1)))
    match = im.draw_matching_features(viz0, cv.KeyPoint_convert(E_match['train_keypoints']),
                                      viz1, cv.KeyPoint_convert(E_match['query_keypoints']))

    cv.namedWindow('Key points 0', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('Key points 0', im.image_size(keys0))
    cv.imshow('Key points 0', keys0)

    cv.namedWindow('Key points 1', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('Key points 1', im.image_size(keys1))
    cv.imshow('Key points 1', keys1)

    cv.namedWindow('Matching points', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('Matching points', im.image_size(match))
    cv.imshow('Matching points', match)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    show_keypoints()


if __name__ == '__main__':
    main()
