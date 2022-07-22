import numpy as np


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
