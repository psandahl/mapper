import cv2 as cv

import os

import mapper.util.external as ext


class KittiData:
    def __init__(self, directory):
        """
        Create a KITTI dataset iterator, with the base directory for the dataset.

        It expects the files calib.txt, poses.txt and directory image_l.
        """
        image_dir = os.path.join(directory, 'image_l')

        self.image_paths = [os.path.join(image_dir, file)
                            for file in sorted(os.listdir(image_dir))]
        self.calib = ext.read_3x4_matrices(
            os.path.join(directory, 'calib.txt'))
        self.poses = ext.read_3x4_matrices(
            os.path.join(directory, 'poses.txt'))

        self.num_data = 0
        self.current_data = 0

        if len(self.calib) > 0 and len(self.poses) == len(self.image_paths):
            self.num_data = len(self.image_paths)
        else:
            print(
                f'Warning: Failed to properly read test data from={directory}')

    def __iter__(self):
        """
        Reset the iteration index.
        """
        self.current_data = 0

        return self

    def __next__(self):
        """
        Get the next item in the dataset.

        Returns:
            Tuple (gray image, 3x4 matrix with calib, pose ground thruth).
        """
        if not self.current_data < self.num_data:
            raise StopIteration

        image = cv.imread(
            self.image_paths[self.current_data], cv.IMREAD_GRAYSCALE)

        retval = (image, self.calib[0], self.poses[self.current_data])

        self.current_data += 1

        return retval
