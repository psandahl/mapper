import unittest
import numpy as np

import unittest

import mapper.image as im
import mapper.keypoint as kp
import mapper.matrix as mat
import mapper.utils as utils


class KeypointTestCase(unittest.TestCase):
    def test_hamming_distance(self):
        desc0 = np.packbits(
            np.array([[[1, 0, 1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0, 1, 0]]]), axis=-1)

        # Differ in four positions.
        desc1 = np.packbits(
            np.array([[[1, 0, 1, 0, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 1, 1]]]), axis=-1)

        self.assertEqual(0, kp.hamming_distance(desc0, desc0))
        self.assertEqual(4, kp.hamming_distance(desc0, desc1))


class ImageTestCase(unittest.TestCase):
    def test_is_image(self):
        self.assertFalse(im.is_image(1))
        self.assertFalse(im.is_image('not image'))
        self.assertFalse(im.is_image(np.array([1, 2, 3, 4, 5, 6])))

        single_channel = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
        self.assertTrue(im.is_image(single_channel))

        dual_channel = np.array([single_channel])
        self.assertTrue(im.is_image(dual_channel))

    def test_num_channels(self):
        single_channel = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
        self.assertEqual(1, im.num_channels(single_channel))

        dual_channel = np.array([single_channel])
        self.assertEqual(2, im.num_channels(dual_channel))

    def test_image_size(self):
        single_channel = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
        self.assertTupleEqual((2, 3), im.image_size(single_channel))

    def test_interpolate_pixel(self):
        img = np.array([0.0, 1.0, 2.0, 3.0]).reshape(2, 2)

        self.assertAlmostEqual(0.0 * 0.0, im.interpolate_pixel(img, 0.0, 0.0))
        self.assertAlmostEqual(0.5 * 1.0, im.interpolate_pixel(img, 0.5, 0.0))
        self.assertAlmostEqual(0.5 * 2.0, im.interpolate_pixel(img, 0.0, 0.5))
        self.assertAlmostEqual(0.25 * 0.0 + 0.25 * 1.0 + 0.25 *
                               2.0 + 0.25 * 3.0, im.interpolate_pixel(img, 0.5, 0.5))

        self.assertAlmostEqual(0.9 * 1.0, im.interpolate_pixel(img, 0.9, 0.0))
        self.assertAlmostEqual(0.9 * 2.0, im.interpolate_pixel(img, 0.0, 0.9))

    def test_to_cv_point(self):
        x, y = im.to_cv_point(np.array([1.1, 1.9]))
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)


class UtilsTestCase(unittest.TestCase):
    def test_fov_from_focal_length(self):
        self.assertEqual(90, utils.fov_from_focal_length(1.0, 2.0))

    def test_focal_length_from_fov(self):
        self.assertTrue(2, utils.focal_length_from_fov(90, 2.0))

    def test_to_and_from_fov(self):
        media_size = 1024

        self.assertAlmostEqual(72.2, utils.fov_from_focal_length(
            utils.focal_length_from_fov(72.2, media_size), media_size))

        self.assertAlmostEqual(45.0, utils.fov_from_focal_length(
            utils.focal_length_from_fov(45.0, media_size), media_size))

        self.assertAlmostEqual(11.123, utils.fov_from_focal_length(
            utils.focal_length_from_fov(11.123, media_size), media_size))

        self.assertAlmostEqual(0.2, utils.fov_from_focal_length(
            utils.focal_length_from_fov(0.2, media_size), media_size))


class MatrixTestCase(unittest.TestCase):
    def test_to_and_from_ideal_intrinsic_matrix(self):
        fov = (30, 20)
        size = (1024, 768)

        imat = mat.ideal_intrinsic_matrix(fov, size)
        fov1, size1 = mat.decomp_intrinsic_matrix(imat)

        self.assertAlmostEqual(fov[0], fov1[0])
        self.assertAlmostEqual(fov[1], fov1[1])
        self.assertTupleEqual(size, size1)


def run_tests():
    unittest.main()


if __name__ == '__main__':
    run_tests()
