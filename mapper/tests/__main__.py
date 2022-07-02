import unittest
import numpy as np

import unittest

import mapper.vision.image as im
import mapper.vision.keypoint as kp
import mapper.vision.matrix as mat
import mapper.vision.transform as tr
import mapper.vision.utils as utils


def assertEqualArray(fw: unittest.TestCase, arr0: np.ndarray,
                     arr1: np.ndarray) -> None:
    fw.assertEqual(len(arr0), len(arr1))
    for i in range(0, len(arr0)):
        fw.assertAlmostEqual(arr0[i], arr1[i])


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

    def test_aspect_ratio(self):
        self.assertAlmostEqual(1024 / 768, utils.aspect_ratio((1024, 768)))


class MatrixTestCase(unittest.TestCase):
    def test_to_and_from_ideal_intrinsic_matrix(self):
        fov = (30, 20)
        size = (1024, 768)

        m = mat.ideal_intrinsic_matrix(fov, size)
        fov1, size1 = mat.decomp_intrinsic_matrix(m)

        self.assertAlmostEqual(fov[0], fov1[0])
        self.assertAlmostEqual(fov[1], fov1[1])
        self.assertTupleEqual(size, size1)

    def test_to_and_from_ypr_matrix_yxz(self):
        yaw = -23.6
        pitch = 7.7
        roll = 167.0

        m = mat.ypr_matrix_yxz(yaw, pitch, roll)
        yaw1, pitch1, roll1 = mat.decomp_ypr_matrix_yxz(m)

        self.assertAlmostEqual(yaw, yaw1)
        self.assertAlmostEqual(pitch, pitch1)
        self.assertAlmostEqual(roll, roll1)

    def test_to_and_from_ypr_matrix_zyx(self):
        yaw = -23.6
        pitch = 7.7
        roll = 167.0

        m = mat.ypr_matrix_zyx(yaw, pitch, roll)
        yaw1, pitch1, roll1 = mat.decomp_ypr_matrix_zyx(m)

        self.assertAlmostEqual(yaw, yaw1)
        self.assertAlmostEqual(pitch, pitch1)
        self.assertAlmostEqual(roll, roll1)

    def test_to_and_from_extrinsic_matrix(self):
        y = 10
        p = -23.9
        r = 125
        t = np.array([10, 20, 30])

        R = mat.ypr_matrix_yxz(y, p, r)
        ext = mat.extrinsic_matrix(R, t)

        R1, t1 = mat.decomp_extrinsic_matrix(ext)
        yy, pp, rr = mat.decomp_ypr_matrix_yxz(R1)

        assertEqualArray(self, t, t1)
        self.assertAlmostEqual(y, yy)
        self.assertAlmostEqual(p, pp)
        self.assertAlmostEqual(r, rr)

    def test_to_and_from_extrinsic_rtvec(self):
        y = 10
        p = -23.9
        r = 125
        t = np.array([10, 20, 30])

        R = mat.ypr_matrix_yxz(y, p, r)
        rvec, tvec = mat.extrinsic_rtvec(R, t)

        R1, t1 = mat.decomp_extrinsic_rtvec(rvec, tvec)
        yy, pp, rr = mat.decomp_ypr_matrix_yxz(R1)

        assertEqualArray(self, t, t1)
        self.assertAlmostEqual(y, yy)
        self.assertAlmostEqual(p, pp)
        self.assertAlmostEqual(r, rr)

    def test_ecef_to_camera_matrix(self):
        m = mat.ecef_to_camera_matrix()

        axis0 = m @ np.array([1.0, 0.0, 0.0])
        assertEqualArray(self, [0.0, 0.0, -1.0], axis0)

        axis1 = m @ np.array([0.0, 1.0, 0.0])
        assertEqualArray(self, [1.0, 0.0, 0.0], axis1)

        axis2 = m @ np.array([0.0, 0.0, 1.0])
        assertEqualArray(self, [0.0, -1.0, 0.0], axis2)

    def test_look_at_yxz(self):
        eye = np.array([0.0, 0.0, 0.0])
        at = np.array([0.0, 0.0, 1.0])
        down = np.array([0.0, 1.0, 0.0])

        R, t = mat.look_at_yxz(eye, at, down)
        assertEqualArray(self, [0, 0, 0], mat.decomp_ypr_matrix_yxz(R))
        assertEqualArray(self, at, t)

        at = np.array([0.0, 1.0, 1.0])
        R, t = mat.look_at_yxz(eye, at, down)
        assertEqualArray(self, [0, -45, 0], mat.decomp_ypr_matrix_yxz(R))
        assertEqualArray(self, at, t)


class TransformTestCase(unittest.TestCase):
    def test_world_to_camera_mat(self):
        goal = np.array([1, 2, 3])  # Always hit camspace pos.

        R = mat.ypr_matrix_yxz(0, 0, 0)
        t = np.array([0, 0, 0])

        # Simple case, no rotation and no translate.
        ext = mat.extrinsic_matrix(R, t)
        xyz_w = np.array([1, 2, 3])
        xyz_c = tr.world_to_camera_mat(ext, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w, tr.camera_to_world_mat(ext, goal))

        # Rotation and translation in camera space.
        R = mat.ypr_matrix_yxz(-90, 0, 0)  # CCW in this case.
        t = np.array([0, 0, -10])
        xyz_w = np.array([-3, 2, -9])
        ext = mat.extrinsic_matrix(R, t)
        xyz_c = tr.world_to_camera_mat(ext, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w, tr.camera_to_world_mat(ext, goal))

        # No rotation and no translation, but in ECEF.
        R = tr.remap_frame(mat.ypr_matrix_zyx(0, 0, 0),
                           mat.ecef_to_camera_matrix())
        t = np.array([0, 0, 0])
        xyz_w = np.array([-3, 1, -2])
        ext = mat.extrinsic_matrix(R, t)
        xyz_c = tr.world_to_camera_mat(ext, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w, tr.camera_to_world_mat(ext, goal))

        # With rotation and translation in ECEF.
        R = tr.remap_frame(mat.ypr_matrix_zyx(90, 0, 0),  # Also CCW.
                           mat.ecef_to_camera_matrix())
        t = np.array([10, 0, 0])
        xyz_w = np.array([9, -3, -2])
        ext = mat.extrinsic_matrix(R, t)
        xyz_c = tr.world_to_camera_mat(ext, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w, tr.camera_to_world_mat(ext, goal))

    def test_world_to_camera_rtvec(self):
        goal = np.array([1, 2, 3])  # Always hit camspace pos.

        R = mat.ypr_matrix_yxz(0, 0, 0)
        t = np.array([0, 0, 0])

        # Simple case, no rotation and no translate.
        rvec, tvec = mat.extrinsic_rtvec(R, t)
        xyz_w = np.array([1, 2, 3])
        xyz_c = tr.world_to_camera_rtvec(rvec, tvec, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w,
                         tr.camera_to_world_rtvec(rvec, tvec, goal))

        # Rotation and translation in camera space.
        R = mat.ypr_matrix_yxz(-90, 0, 0)  # CCW in this case.
        t = np.array([0, 0, -10])
        rvec, tvec = mat.extrinsic_rtvec(R, t)
        xyz_w = np.array([-3, 2, -9])
        xyz_c = tr.world_to_camera_rtvec(rvec, tvec, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w,
                         tr.camera_to_world_rtvec(rvec, tvec, goal))

        # No rotation and no translation, but in ECEF.
        R = tr.remap_frame(mat.ypr_matrix_zyx(0, 0, 0),
                           mat.ecef_to_camera_matrix())
        t = np.array([0, 0, 0])
        rvec, tvec = mat.extrinsic_rtvec(R, t)
        xyz_w = np.array([-3, 1, -2])
        xyz_c = tr.world_to_camera_rtvec(rvec, tvec, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w,
                         tr.camera_to_world_rtvec(rvec, tvec, goal))

        # With rotation and translation in ECEF.
        R = tr.remap_frame(mat.ypr_matrix_zyx(90, 0, 0),  # Also CCW.
                           mat.ecef_to_camera_matrix())
        t = np.array([10, 0, 0])
        rvec, tvec = mat.extrinsic_rtvec(R, t)
        xyz_w = np.array([9, -3, -2])
        xyz_c = tr.world_to_camera_rtvec(rvec, tvec, xyz_w)
        assertEqualArray(self, goal, xyz_c)
        assertEqualArray(self, xyz_w,
                         tr.camera_to_world_rtvec(rvec, tvec, goal))


def run_tests():
    unittest.main()


if __name__ == '__main__':
    run_tests()
