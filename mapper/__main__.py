import cv2 as cv

import mapper.image as im
import mapper.keypoint as kp


def show_keypoints():
    gray0 = im.scale_image(im.gray_convert(
        im.read_image_bgr('c:/Users/patsa/Pictures/seq4/IMG_0124.jpeg')), 0.5)
    gray1 = im.scale_image(im.gray_convert(
        im.read_image_bgr('c:/Users/patsa/Pictures/seq4/IMG_0125.jpeg')), 0.5)

    points0 = kp.ssc_refine(kp.detect(gray0), 1000, im.image_size(gray0))
    points1 = kp.ssc_refine(kp.detect(gray1), 1000, im.image_size(gray1))

    train = kp.compute(gray0, points0)
    query = kp.compute(gray1, points1)

    first_matches = kp.match(train, query)
    viz0 = im.visualization_image(gray0)
    viz1 = im.visualization_image(gray1)

    match0 = im.draw_matching_features(viz0, cv.KeyPoint_convert(first_matches['train_keypoints']),
                                       viz1, cv.KeyPoint_convert(first_matches['query_keypoints']))

    cv.namedWindow('First matches', cv.WINDOW_NORMAL +
                   cv.WINDOW_KEEPRATIO + cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow('First matches', im.image_size(match0))
    cv.imshow('First matches', match0)

    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    show_keypoints()


if __name__ == '__main__':
    main()
