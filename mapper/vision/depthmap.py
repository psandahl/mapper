
import cv2 as cv
import numpy as np

import mapper.vision.image as im
import mapper.vision.matrix as mat
import mapper.vision.transform as trf
import mapper.vision.utils as utils


class Gaussian():
    def __init__(self):
        """
        Create a new gaussian distribution.
        """
        self.num = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.mean = 0.0
        self.variance = 0.0

    def add(self, value: float) -> None:
        """
        Add a new number to the distribution, and update num, sum, mean and variance.

        Parameters:
            value: A number to add.
        """
        self.num += 1
        self.sum += value
        self.sumsq += value ** 2
        self.mean = self.sum / self.num

        meansq = self.mean ** 2

        self.variance = self.sumsq / self.num - meansq

    def ratio(self) -> float:
        if self.num > 0:
            return self.variance / self.mean
        else:
            return 0.0


class DepthMap():
    def __init__(self):
        """
        Create a new depth map.
        """
        self.map = dict()

    def add(self, px: tuple, depth: float) -> None:
        """
        Add a new depth value to the map.
        """
        assert isinstance(px, tuple)
        assert len(px) == 2
        assert isinstance(depth, float)

        if px in self.map:
            self.map[px].add(depth)
        else:
            gaussian = Gaussian()
            gaussian.add(depth)
            self.map[px] = gaussian

    def num_map_points(self):
        """
        Return the number of map points.
        """
        return len(self.map)

    def purge(self, threshold: float) -> None:
        """
        Purge bad apples from a depth map.
        """
        before = len(self.map)

        for key in list(self.map.keys()):
            if self.map[key].ratio() > threshold:
                del self.map[key]

        after = len(self.map)

        print(f' purge depth map. Of total={before}, inliers kept={after}')

    def export_to_projection(self,
                             src_inv_intrinsic: np.ndarray,
                             src_inv_extrinsic: np.ndarray,
                             tgt_intrinsic: np.ndarray,
                             tgt_extrinsic: np.ndarray,
                             tgt_image_size: tuple):
        """
        Export this depth map to a new depth map, but with
        another projection.
        """
        assert isinstance(src_inv_intrinsic, np.ndarray)
        assert src_inv_intrinsic.shape == (3, 3)
        assert isinstance(src_inv_extrinsic, np.ndarray)
        assert src_inv_extrinsic.shape == (3, 4)
        assert isinstance(tgt_intrinsic, np.ndarray)
        assert tgt_intrinsic.shape == (3, 3)
        assert isinstance(tgt_extrinsic, np.ndarray)
        assert tgt_extrinsic.shape == (3, 4)
        assert isinstance(tgt_image_size, tuple)
        assert len(tgt_image_size) == 2

        w, h = tgt_image_size

        tgt_depth_map = DepthMap()

        tgt_projection = mat.projection_matrix(tgt_intrinsic, tgt_extrinsic)
        for src_px, gaussian in self.map.items():
            # print(
            #    f'depth item. count={gaussian.num}, mean={gaussian.mean}, var={gaussian.variance}')

            # Reconstruct the xyz position in the src projection.

            # TODO: Filter on variance.
            xyz = trf.unproject_point(
                src_inv_intrinsic, src_inv_extrinsic, src_px, gaussian.mean)

            # Make outlier filtering (infront of camera, and inside of image).
            if trf.infront_of_camera(tgt_extrinsic, xyz):
                tgt_px, depth = trf.project_point(tgt_projection, xyz)
                tgt_u, tgt_v = tgt_px

                if tgt_u >= 0.0 and tgt_u < w and tgt_v >= 0.0 and tgt_v < h:
                    tgt_depth_map.add((tgt_u, tgt_v), depth)

        return tgt_depth_map

    def export_to_visualization_image(self, image: np.ndarray,
                                      max_depth: float = 100.0) -> None:
        """
        Export this depth map to a provided visualization image.
        """
        assert im.is_image(image)
        assert im.num_channels(image) == 3

        for px, gaussian in self.map.items():
            bgr = utils.depth_to_bgr(gaussian.mean, max_depth)
            u, v = np.round(px).astype(int)
            cv.circle(image, (u, v), 1, bgr, -1, cv.LINE_AA)
