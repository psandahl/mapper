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

    features0 = im.generate_features(gray0)
    im.draw_features(viz0, features0)

    flow = im.dense_optical_flow(gray0, gray1)
    flow_viz = im.gray_flow_visualization_image(flow)

    cv.namedWindow('Image 0 with features', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Image 0 with features', viz0)

    cv.namedWindow('Image 1 with features', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Image 1 with features', viz1)

    cv.namedWindow('Gray flow image', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.imshow('Gray flow image', flow_viz)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    show_flow()


if __name__ == '__main__':
    main()
