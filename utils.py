import numpy as np
import sympy

def is_square(mat):
    if not (isinstance(mat, np.ndarray) or isinstance(mat, sympy.MatrixBase)):
        raise TypeError("Expected an np.ndarray or sympy.MatrixBase instance")

    shape = mat.shape
    return shape[0] == shape[1]
