import cv2 as cv
import numpy as np

import mapper.image as im
import mapper.keypoint as kp


def show_keypoints():
    gray0 = im.scale_image(im.gray_convert(
        im.read_image_bgr('c:/Users/patri/bilder/IMG_0145.jpeg')), 0.5)
    gray1 = im.scale_image(im.gray_convert(
        im.read_image_bgr('c:/Users/patri/bilder/IMG_0146.jpeg')), 0.5)

    points0 = kp.refine(kp.detect(gray0), 700, im.image_size(gray0))
    points1 = kp.refine(kp.detect(gray1), 700, im.image_size(gray1))

    viz0 = im.visualization_image(gray0)
    im.draw_features(viz0, cv.KeyPoint_convert(points0))
    viz1 = im.visualization_image(gray1)
    im.draw_features(viz1, cv.KeyPoint_convert(points1))

    cv.namedWindow('Features 0', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('Features 0', im.image_size(viz0))
    cv.imshow('Features 0', viz0)

    cv.namedWindow('Features 1', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('Features 1', im.image_size(viz1))
    cv.imshow('Features 1', viz1)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    show_keypoints()


if __name__ == '__main__':
    main()
