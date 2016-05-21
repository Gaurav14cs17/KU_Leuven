import numpy as np

from asm.pca import PCAModel
from asm.shape import AlignedShapeList, Shape


class ActiveShapeModel:
    """ An Active Shape Model based on
    Cootes, Tim, E. R. Baldock, and J. Graham.
    "An introduction to active shape models."
    Image processing and analysis (2000): 223-248.

    Attributes:
    _aligned_shapes An AlignedShapeList containing
    the training shapes
    _model  The underlying PCA Model
    _modes The number of modes of the model
    _bmax The limits of variation of the shape model

    Authors: David Torrejon and Bharath Venkatesh

"""

    def __init__(self, shapes, pca_variance_captured=0.9, gpa_tol=1e-7, gpa_max_iters=10000):
        """
        Constructs the Active Shape Model based on the given list of Shapes.
        :param shapes: A list of Shapes
        :param pca_variance_captured: The fraction of variance to be captured by the shape model
        :param gpa_tol: tol: The convergence threshold for gpa
        (Default: 1e-7)
        :param gpa_max_iters: The maximum number of iterations
        permitted for gpa (Default: 10000)
        """
        self._aligned_shapes = AlignedShapeList(shapes, gpa_tol, gpa_max_iters)
        self._model = PCAModel(self._aligned_shapes.raw())
        self._modes = self._model.k_cutoff(pca_variance_captured)
        self._b_max = 3 * self._model.eigenvalues()[0:self._modes] ** 0.5

    def aligned_shapes(self):
        return self._aligned_shapes.shapes()

    def mean_shape(self):
        """
        Returns the mean shape of the model
        :return: A Shape object containing the mean shape
        """
        return self._aligned_shapes.mean_shape()

    def modes(self):
        """
        Returns the number of modes of the model
        :return: the number of modes
        """
        return self._modes

    def generate_shape(self, factors):
        """
        Generates a shape based on a vector of factors of size
        equal to the number of modes of the model, with element
        values between -1 and 1
        :param factors: A vector of size modes() with values
        between -1 and 1
        :return: A Shape object containing the generated shape
        """
        p = self._model.eigenvectors()[:, 0:self._modes]
        pb = np.dot(p, factors * self._b_max)
        return Shape(self.mean_shape().raw() + Shape.from_collapsed_shape(pb).raw())

    def mode_shapes(self, m):
        """
        Returns the modal shapes of the model (Variance limits)
        :param m: A list of Shape objects containing the modal shapes
        """
        if m < 0 or m >= self._modes:
            raise ValueError('Number of modes must be within [0,modes()-1]')
        factors = np.zeros(self._modes)
        mode_shapes = []
        for i in range(-1, 2):
            factors[m] = i
            mode_shapes.append(self.generate_shape(factors))
        return mode_shapes
