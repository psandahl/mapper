import cv2 as cv
import numpy as np

from enum import Enum
import math

import mapper.vision.image as im


class KeypointType(Enum):
    AGAST = 1
    AKAZE = 2
    ORB = 3


detector = None
extractor = None
matcher = None


def configure_keypoint(keypoint_type: KeypointType,
                       agast_threshold: int = 15,
                       orb_features: int = 5000) -> None:
    """
    Configure keypoint detector/matcher.
    """
    global detector
    global extractor
    global matcher

    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6, key_size=12, multi_probe_level=1)
    # search_params = dict(checks=50)
    # matcher = cv.FlannBasedMatcher(indexParams=index_params,
    #                                searchParams=search_params)

    if keypoint_type == KeypointType.AGAST:
        detector = cv.AgastFeatureDetector_create()
        extractor = cv.xfeatures2d.BriefDescriptorExtractor_create()

        detector.setThreshold(agast_threshold)
        detector.setNonmaxSuppression(False)
    elif keypoint_type == KeypointType.AKAZE:
        detector = cv.AKAZE_create()
        extractor = detector
    elif keypoint_type == KeypointType.ORB:
        detector = cv.ORB_create(orb_features)
        extractor = detector
    else:
        print('Error: configure_keypoint() cannot determine type')


def detect(image: cv.Mat, num_ret_points: int = 1500) -> list:
    """
    Detect keypoints in a gray scale image using the configured detector. Will
    be refined using SSC.

    Parameters:
        image: Image to detect keypoints in.
        num_ret_points: The number of wanted points to return (approx).

    Returns:
        Keypoints for the image.
    """
    assert im.is_image(image)
    assert im.num_channels(image) == 1

    if detector is None:
        configure_keypoint(KeypointType.AGAST)

    assert not detector is None
    assert not extractor is None
    assert not matcher is None

    return SSC_refine(detector.detect(image), num_ret_points, im.image_size(image))


def SSC_refine(keypoints: tuple, num_ret_points: int, image_size: tuple, tolerance: float = 0.1) -> list:
    """
    Refine keypoints to select the strongest points, distributed
    over the image.

    This is the supression via square covering (SSC) algorithm from
    https://github.com/BAILOOL/ANMS-Codes

    Parameters:
        keypoints: Detected keypoints.
        num_ret_points: The number of wanted points to return (approx).
        image_size: Tuple (width, height) of image.
        tolerance: Algorithm parameter.

    Returns:
        The selected keypoints.
    """
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    cols, rows = image_size

    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = (
        sol1 if (sol1 > sol2) else sol2
    )  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if (
            width == prev_width or low > high
        ):  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [
            [False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)
        ]
        result = []

        for i in range(len(keypoints)):
            row = int(
                math.floor(keypoints[i].pt[1] / c)
            )  # get position of the cell current point is located at
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int(
                    (row - math.floor(width / c))
                    if ((row - math.floor(width / c)) >= 0)
                    else 0
                )
                row_max = int(
                    (row + math.floor(width / c))
                    if ((row + math.floor(width / c)) <= num_cell_rows)
                    else num_cell_rows
                )
                col_min = int(
                    (col - math.floor(width / c))
                    if ((col - math.floor(width / c)) >= 0)
                    else 0
                )
                col_max = int(
                    (col + math.floor(width / c))
                    if ((col + math.floor(width / c)) <= num_cell_cols)
                    else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            # cover cells within the square bounding box with width w
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    for i in range(len(result_list)):
        selected_keypoints.append(keypoints[result_list[i]])

    return selected_keypoints


def compute(image: cv.Mat, keypoints: list) -> tuple:
    """
    Compute descriptors for a set of keypoints.

    Parameters:
        image: Image for which the descriptors are to be computed.
        keypoints: List or tuple with keypoints.

    Returns:
        Tuple (keypoints, descriptors).
    """
    assert im.is_image(image)
    assert im.num_channels(image)
    assert isinstance(keypoints, tuple) or isinstance(keypoints, list)

    return extractor.compute(image, keypoints)


def match(train: tuple, train_id: int, query: tuple, query_id: int) -> dict:
    """
    Match descriptors and select keypoints and descriptors from best matches.

    Parameters:
        train: Tuple (keypoints, descriptors).
        query: Tuple (keypoints, descriptors).

    Returns:
        A dictionary with keys: train_keypoints, train_descriptors,
                                query_keypoints, query_descriptors
    """
    kpt0, desc0 = train
    kpt1, desc1 = query

    raw_matches = matcher.knnMatch(desc0, desc1, 2)

    kpt00 = list()
    desc00 = list()
    kpt11 = list()
    desc11 = list()
    for m, n in raw_matches:
        if m.distance < n.distance * 0.8:  # Lowe's ratio.
            # Yes, this is a bit weird. Train index into 1 and
            # query index into 0. But it is how it is.

            # https://stackoverflow.com/questions/13318853/opencv-drawmatches-queryidx-and-trainidx/13320083#13320083
            train_idx = m.trainIdx
            query_idx = m.queryIdx

            kpt00.append(kpt0[query_idx])
            desc00.append(desc0[query_idx])
            kpt11.append(kpt1[train_idx])
            desc11.append(desc1[train_idx])

    match = dict()
    match['train_id'] = train_id
    match['train_keypoints'] = kpt00
    match['train_descriptors'] = desc00
    match['query_id'] = query_id
    match['query_keypoints'] = kpt11
    match['query_descriptors'] = desc11

    return match


def hamming_distance(desc0, desc1):
    """
    Compute the Hamming distance between two binary descriptors.
    """
    return np.unpackbits((desc0 ^ desc1).view('uint8')).sum()
