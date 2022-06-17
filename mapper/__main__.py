import cv2 as cv
import numpy as np

import mapper.image as im


def show_images():
    bgr = im.read_image_bgr('c:/Users/patri/bilder/left.JPG')
    gray = im.gray_convert(bgr)
    gray2 = im.gray_convert(gray)

    cv.namedWindow('show', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)

    cv.setWindowTitle('show', 'Original BGR')
    cv.imshow('show', bgr)
    key = cv.waitKey(0)

    cv.setWindowTitle('show', 'Gray from BGR')
    cv.imshow('show', gray)
    cv.waitKey(0)

    cv.setWindowTitle('show', 'Gray from gray')
    cv.imshow('show', gray2)
    cv.waitKey(0)

    cv.destroyWindow('show')


def main():
    print('hello')


if __name__ == '__main__':
    main()
