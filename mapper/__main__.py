import cv2 as cv
import numpy as np

import mapper.image as im
import mapper.keypoint as kp


def show_dense_flow():
    bgr0 = im.read_image_bgr('c:/Users/patri/bilder/IMG_0161.jpeg')
    gray0 = im.gray_convert(bgr0)
    viz0 = im.visualization_image(gray0)

    bgr1 = im.read_image_bgr('c:/Users/patri/bilder/IMG_0162.jpeg')
    gray1 = im.gray_convert(bgr1)
    viz1 = im.visualization_image(gray1)

    flow = im.dense_optical_flow(gray0, gray1)
    #flow_viz = im.gray_flow_visualization_image(flow)

    features0 = im.generate_features(gray0)
    features1 = im.matching_features_from_flow(flow, features0)

    H, features00, features11 = im.find_homography(features0, features1)

    print(f'features0={len(features0)}, features00={len(features00)}')

    match = im.draw_matching_features(
        viz0, features00, viz1, features11, step=2)

    # cv.namedWindow('Gray flow image', cv.WINDOW_NORMAL +
    #               cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    #cv.imshow('Gray flow image', flow_viz)

    cv.namedWindow('Matching features', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Matching features', match)

    cv.waitKey(0)
    cv.destroyAllWindows()


def show_keypoints():
    bgr0 = im.read_image_bgr('c:/Users/patri/bilder/IMG_0145.jpeg')
    gray0 = im.gray_convert(bgr0)

    viz0 = im.visualization_image(gray0)
    viz1 = im.visualization_image(gray0)

    points0 = kp.detect(gray0)
    features0 = cv.KeyPoint_convert(points0)

    points1 = kp.refine(points0, 500, im.image_size(gray0))
    features1 = cv.KeyPoint_convert(points1)

    print(f'Num features={len(features0)}')
    print(f'Num filtered features={len(features1)}')

    im.draw_features(viz0, features0)
    im.draw_features(viz1, features1)

    cv.namedWindow('Features0', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Features0', viz0)

    cv.namedWindow('Features1', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Features1', viz1)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    # show_dense_flow()
    show_keypoints()


if __name__ == '__main__':
    main()
