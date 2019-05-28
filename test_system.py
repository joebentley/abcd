import pytest
import numpy as np
import sympy
from system import System, DimensionError

a, b, c = sympy.symbols('a b c')


def test_instantiating_with_wrong_type_raises_type_error():
    with pytest.raises(TypeError):
        System(1, 2, 3, "a")
    with pytest.raises(TypeError):
        System(1, 2, "b", "a")


def test_instantiating_with_correct_type_does_not_raises_type_error():
    System(1, 2, 3, 4)
    mat = np.array([[1, 2], [3, 4]])
    sys = System(mat, mat, mat, mat)
    assert not sys.isSymbolic
    with pytest.raises(DimensionError):
        System(np.array(mat), 3, 2, 5)
    # testing symbolics
    System(a, b, c)
    mat = sympy.Matrix([[1, 0], [0, 1]])
    System(mat, mat, mat, mat)


def test_instantiating_with_correct_matrix_dimensions_does_not_raise_dimension_error():
    amat = np.array([[1, 2], [3, 4]])
    System(amat, amat, amat, amat)
    bmat = np.array([[1], [2]])
    System(amat, bmat, np.transpose(bmat), 1)

    amat = sympy.Matrix(amat.tolist())
    bmat = sympy.Matrix(bmat.tolist())
    System(amat, amat, amat, amat)
    System(amat, bmat, np.transpose(bmat), 1)


def test_instantiating_with_wrong_matrix_dimensions_raises_dimension_error():
    amat = np.array([[1, 2], [3, 4]])
    bmat = np.array([[1], [2]])

    with pytest.raises(DimensionError):
        System(np.array([[1, 2], [3, 4], [5, 6]]), bmat, np.transpose(bmat), 1)
    with pytest.raises(DimensionError):
        System(sympy.Matrix([[1, 2], [3, 4], [5, 6]]),
               sympy.Matrix(bmat.tolist()), sympy.Matrix(bmat.tolist()).transpose(), 1)

    with pytest.raises(DimensionError):
        System(amat, bmat, bmat, 1)
    with pytest.raises(DimensionError):
        System(sympy.Matrix(amat.tolist()), sympy.Matrix(bmat.tolist()), sympy.Matrix(bmat.tolist()), 1)

    # will assume other dimensions are checked properly for sympy
    with pytest.raises(DimensionError):
        System(amat, bmat, np.transpose(bmat), np.array([[1], [2]]))
    with pytest.raises(DimensionError):
        System(amat, np.transpose(bmat), bmat, 1)
    with pytest.raises(DimensionError):
        System(amat, np.array([[1, 2], [3, 4]]), np.transpose(bmat), 1)


def test_instantiating_with_any_symbolic_variables_sets_symbolic_flag():
    sys = System(a, b, c)
    assert sys.isSymbolic
    mat = sympy.Matrix([[1, 0], [0, 1]])
    sys = System(mat, mat, mat, mat)
    assert sys.isSymbolic
    # inputs should not be set
    assert sys.inputs is None


def test_direct_feed_matrix_defaults_to_appropriate_identity_matrix():
    sys = System(1, 2, 3)
    assert np.array_equal(sys.d, np.identity(1))

    sys = System(a, b, c)
    assert sys.d.equals(sympy.eye(1))

    # 2 states, 2 inputs, 2 outputs
    mat = np.array([[1, 2], [3, 4]])
    bc = np.array([[0, 1], [1, 0]])
    sys = System(mat, bc, bc)
    assert np.array_equal(sys.d, np.identity(2))

    mat = sympy.Matrix(mat.tolist())
    bc = sympy.Matrix(bc.tolist())
    sys = System(mat, bc, bc)
    assert sys.d.equals(sympy.eye(2))


def test_direct_feed_matrix_if_num_inputs_and_outputs_dont_match():
    # 2 states, 1 input, 2 outputs
    with pytest.raises(DimensionError):
        System(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]), np.array([[1, 2], [3, 4]]))
    with pytest.raises(DimensionError):
        System(sympy.Matrix([[1, 2], [3, 4]]), sympy.Matrix([[1], [2]]), sympy.Matrix([[1, 2], [3, 4]]))


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


def test_set_input_and_step_evolution_should_raise_type_error_if_symbolic():
    sys = System(a, b, c)
    with pytest.raises(TypeError):
        sys.set_input(lambda dt: dt)
    with pytest.raises(TypeError):
        sys.step_evolution(0)

def test_j_matrix_as_expected():
    pass