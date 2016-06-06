import cv2
import numpy as np
import math

from asm.pca import ModedPCAModel


class GreyModel:
    """ A grey level point model based on
        Cootes, Timothy F., and Christopher J. Taylor.
         "Active Shape Model Search using Local Grey-Level Models:
         A Quantitative Evaluation." BMVC. Vol. 93. 1993.

            Attributes:
                _models: A list of ModedPCA models of the grey levels
                         of the landmark points


            Authors: David Torrejon and Bharath Venkatesh

    """

    def _get_point_normal_pixel_coordinates(self, shape, point_index):
        """
        Get the coordinates of pixels lying on the normal of the point
        :param shape:
        :param point_index:
        :return:
        """
        point = shape.get_point(point_index)
        neighborhood = shape.get_neighborhood(point_index, self._normal_neighborhood)
        line = cv2.fitLine(neighborhood, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = line[0:2] / math.sqrt(np.sum(line[0:2] ** 2))
        return [[int(point[1] + (incr * slope[0]) + 0.5), int(point[0] - (incr * slope[1]) + 0.5)] for incr in
                range(-self._number_of_pixels, self._number_of_pixels + 1)]

    def _get_normal_grey_levels_for_single_point_single_image(self, image, shape, point_index):
        coordinate_list = self._get_point_normal_pixel_coordinates(shape, point_index)
        data = np.zeros((2 * self._number_of_pixels + 1, 1), dtype=float)
        ctr = 0
        h, w = image.shape
        for coordinates in coordinate_list:
            if 0 <= coordinates[0] < h and 0 <= coordinates[1] < w:
                data[ctr] = image[coordinates[0]][coordinates[1]]
            ctr += 1
        return data

    def getModel(self, point_index):
        return self._models[point_index]

    def generate_grey(self, point_index, factors):
        """
        Generates a grey vector based on a vector of factors of size
        equal to the number of modes of the model, with element
        values between -1 and 1
        :param point_index: The index of the landmark
        :param factors: A vector of size modes() with values
        between -1 and 1
        :return: A vector containing the grey levels of a point
        """
        return self._models[point_index].mean() + self._models[point_index].generate_deviation(factors)

    def mode_greys(self, point_index,m):
        """
        Returns the modal grey levels of the model (Variance limits)
        :param m: A list of vectors containing the modal greys
        """
        if m < 0 or m >= self.modes():
            raise ValueError('Number of modes must be within [0,modes()-1]')
        factors = np.zeros(self._models[point_index].modes())
        mode_greys = []
        for i in range(-1, 2):
            factors[m] = i
            mode_greys.append(self.generate_grey(point_index,factors))
        return mode_greys

    def __init__(self, images, shapes, number_of_pixels=5, pca_variance_captured=0.9, normal_point_neighborhood=4):
        self._normal_neighborhood = normal_point_neighborhood
        self._number_of_pixels = number_of_pixels
        self._models = []
        for i in range(shapes[0].size()):
            plist = []
            for j in range(len(images)):
                plist.append(self._get_normal_grey_levels_for_single_point_single_image(images[j], shapes[j], i))
            self._models.append(ModedPCAModel(np.array(plist), pca_variance_captured))
