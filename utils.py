import numpy as np


def is_square(mat):
    if not isinstance(mat, np.ndarray):
        raise TypeError("Expected an np.ndarray instance")

    shape = mat.shape
    return shape[0] == shape[1]
