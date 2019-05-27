import pytest
import numpy as np
import utils


def test_block_matrix():
    pass


def test_is_matrix_square():
    assert utils.is_square(np.array([[1, 2], [3, 4]]))
    assert not utils.is_square(np.array([[1, 2], [3, 4], [5, 6]]))

    with pytest.raises(TypeError):
        utils.is_square("hello")
