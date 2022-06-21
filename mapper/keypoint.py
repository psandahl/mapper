import cv2 as cv

import math

import mapper.image as im

detector = cv.AgastFeatureDetector_create()
extractor = cv.xfeatures2d.BriefDescriptorExtractor_create()


def detect(image: cv.Mat, threshold=15) -> tuple:
    """
    Detect keypoints in a gray scale image using AGAST.

    Parameters:
        image: Image to detect keypoints in.
        threshold: AGAST threshold (lower = more points).

    Returns:
        Keypoints for the image.
    """
    assert im.is_image(image), 'Argument is assumed to be an image'
    assert im.num_channels(image) == 1, 'Image is assumed to be gray scale'

    detector.setThreshold(threshold)
    detector.setNonmaxSuppression(False)

    return detector.detect(image)


def refine(keypoints: tuple, num_ret_points: int, image_size: tuple, tolerance: float = 0.1) -> list:
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


def compute(image: cv.Mat, keypoints: any) -> tuple:
    """
    Compute descriptors for a set of keypoints.

    Parameters:
        image: Image for which the descriptors are to be computed.
        keypoints: List or tuple with keypoints.

    Returns:
        Tuple (keypoints, descriptors).
    """
    assert im.is_image(image), 'Argument is assumed to be an image'
    assert im.num_channels(image), 'Image is supposed to be a gray scale image'
    assert isinstance(keypoints, tuple) or isinstance(
        keypoints, list), 'Tuple or list'

    return detector.compute(image, keypoints)
