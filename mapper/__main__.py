import cv2 as cv
import numpy as np

import mapper.image as im


def show_images():
    bgr = im.read_image_bgr('c:/Users/patri/bilder/IMG_0111.jpeg')

    gray = im.gray_convert(bgr)
    viz = im.visualization_image(gray)

    features = im.generate_features(gray)
    im.draw_features(viz, features)

    cv.namedWindow('show', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)

    cv.setWindowTitle('show', 'Features')
    cv.imshow('show', viz)
    cv.waitKey(0)

    cv.destroyWindow('show')


def show_flow():
    bgr0 = im.read_image_bgr('c:/Users/patri/bilder/IMG_0141.jpeg')
    gray0 = im.gray_convert(bgr0)
    viz0 = im.visualization_image(gray0)

    bgr1 = im.read_image_bgr('c:/Users/patri/bilder/IMG_0142.jpeg')
    gray1 = im.gray_convert(bgr1)
    viz1 = im.visualization_image(gray1)

    flow = im.dense_optical_flow(gray0, gray1)
    #flow_viz = im.gray_flow_visualization_image(flow)

    features0 = im.generate_features(gray0)
    features1 = im.matching_features_from_flow(flow, features0)

    H, features00, features11 = im.find_homography(features0, features1)

    match = im.draw_matching_features(viz0, features00, viz1, features11)

    # cv.namedWindow('Gray flow image', cv.WINDOW_NORMAL +
    #               cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    #cv.imshow('Gray flow image', flow_viz)

    cv.namedWindow('Matching features', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Matching features', match)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    show_flow()


if __name__ == '__main__':
    main()
