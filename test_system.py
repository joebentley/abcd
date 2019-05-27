import pytest
import numpy as np
from system import System, DimensionError


def test_instantiating_with_wrong_type_raises_type_error():
    with pytest.raises(TypeError):
        System(1, 2, 3, "a")
    with pytest.raises(TypeError):
        System(1, 2, "b", "a")


def test_instantiating_with_correct_type_does_not_raises_type_error():
    System(1, 2, 3, 4)
    mat = np.array([[1, 2], [3, 4]])
    System(mat, mat, mat, mat)
    with pytest.raises(DimensionError):
        System(np.array(mat), 3, 2, 5)


def test_instantiating_with_correct_matrix_dimensions_does_not_raise_dimension_error():
    amat = np.array([[1, 2], [3, 4]])
    System(amat, amat, amat, amat)
    bmat = np.array([[1], [2]])
    System(amat, bmat, np.transpose(bmat), 1)


def test_instantiating_with_wrong_matrix_dimensions_raises_dimension_error():
    amat = np.array([[1, 2], [3, 4]])
    bmat = np.array([[1], [2]])
    with pytest.raises(DimensionError):
        System(np.array([[1, 2], [3, 4], [5, 6]]), bmat, np.transpose(bmat), 1)
    with pytest.raises(DimensionError):
        System(amat, bmat, bmat, 1)
    with pytest.raises(DimensionError):
        System(amat, bmat, np.transpose(bmat), np.array([[1], [2]]))
    with pytest.raises(DimensionError):
        System(amat, np.transpose(bmat), bmat, 1)
    with pytest.raises(DimensionError):
        System(amat, np.array([[1, 2], [3, 4]]), np.transpose(bmat), 1)


def test_direct_feed_matrix_defaults_to_identity():
    sys = System(1, 2, 3)
    assert sys.d == 1

    # 2 states, 2 inputs, 2 outputs
    mat = np.array([[1, 2], [3, 4]])
    bc = np.array([[0, 1], [1, 0]])
    sys = System(mat, bc, bc)
    assert np.array_equal(sys.d, np.identity(2))


def test_direct_feed_matrix_if_num_inputs_and_outputs_dont_match():
    # 2 states, 1 input, 2 outputs
    with pytest.raises(DimensionError):
        System(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]), np.array([[1, 2], [3, 4]]))


def test_getting_and_setting_state():
    amat = np.array([[1, 2], [3, 4]])
    bmat = np.array([[1], [2]])
    sys = System(amat, bmat, np.transpose(bmat), 1)
    assert np.array_equal(sys.state, np.array([[0], [0]]))

    sys.state = np.array([[1], [0]])
    assert np.array_equal(sys.state, np.array([[1], [0]]))

    with pytest.raises(DimensionError):
        sys.state = np.array([1, 0])

    sys.state = 100
    assert np.array_equal(sys.state, np.array([[100], [100]]))

    sys = System(1, 1, 1, 1)
    assert np.array_equal(sys.state, np.array([[0]]))
    sys.state = 100
    assert np.array_equal(sys.state, np.array([[100]]))


def test_time_steps_properly():
    sys = System(-1, 1, 1, 1)
    time_series = np.linspace(0, 1, num=1000)
    dt = np.diff(time_series)[0]

    for _ in time_series:
        sys.step_evolution(dt)

    assert pytest.approx(sys.time_elapsed - max(time_series) + dt, 1e-8)
