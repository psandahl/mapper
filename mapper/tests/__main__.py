import unittest
import numpy as np

import unittest

import mapper.image as im


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


def run_tests():
    unittest.main()


if __name__ == '__main__':
    run_tests()
