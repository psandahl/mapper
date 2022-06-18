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
    key = cv.waitKey(0)

    cv.destroyWindow('show')


def main():
    show_images()


if __name__ == '__main__':
    main()
