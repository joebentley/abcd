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
    assert not sys.is_symbolic
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
    assert sys.is_symbolic
    mat = sympy.Matrix([[1, 0], [0, 1]])
    sys = System(mat, mat, mat, mat)
    assert sys.is_symbolic
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


def test_j_matrix_throws_error_if_there_are_not_pairs_of_states():
    sys = System(a, b, c)
    with pytest.raises(DimensionError):
        sys.j_matrix
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sys = System(mat, mat, mat)
    with pytest.raises(DimensionError):
        sys.j_matrix


def test_t_matrix_throws_error_if_there_are_not_pairs_of_inputs():
    sys = System(a, b, c)
    with pytest.raises(DimensionError):
        sys.quantum_t_matrix
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sys = System(mat, mat, mat)
    with pytest.raises(DimensionError):
        sys.quantum_t_matrix


def test_j_matrix_gives_correct_type_and_form():
    mat = np.identity(2)
    sys = System(mat, mat, mat)

    j_mat = sys.j_matrix
    assert isinstance(j_mat, np.ndarray)
    assert np.array_equal(j_mat, np.diag([1, -1]))

    mat = np.identity(4)
    sys = System(mat, mat, mat)
    assert np.array_equal(sys.j_matrix, np.diag([1, -1, 1, -1]))

    mat = sympy.eye(4)
    sys = System(mat, mat, mat, mat)
    j_mat = sys.j_matrix
    assert isinstance(j_mat, sympy.MatrixBase)
    assert j_mat.equals(sympy.diag(*[1, -1, 1, -1]))


def test_t_matrix_gives_correct_type_and_form():
    mat = np.identity(2)
    bmat = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    sys = System(mat, bmat, mat, bmat)

    t_mat = sys.quantum_t_matrix
    assert isinstance(t_mat, np.ndarray)
    assert np.array_equal(t_mat, np.diag([1, -1, 1, -1]))

    mat = sympy.Matrix(mat.tolist())
    bmat = sympy.Matrix(bmat.tolist())
    sys = System(mat, bmat, mat, bmat)

    t_mat = sys.quantum_t_matrix
    assert isinstance(t_mat, sympy.MatrixBase)
    assert t_mat.equals(sympy.diag(*[1, -1, 1, -1]))


def test_physical_realizability():
    # this system is the unstable filter
    a_mat = sympy.Matrix([[2, 0], [0, 2]])
    b_mat = d_mat = sympy.eye(2)
    c_mat = sympy.Matrix([[4, 0], [0, 4]])

    a_mat_prime = sympy.Matrix([[2, 0], [0, 2]])
    b_mat_prime = 2 * sympy.Matrix([[0, 1], [-1, 0]])
    c_mat_prime = 2 * sympy.Matrix([[0, -1], [1, 0]])
    d_mat_prime = d_mat

    assert not System(a_mat, b_mat, c_mat, d_mat).is_physically_realizable
    assert System(a_mat_prime, b_mat_prime, c_mat_prime, d_mat_prime).is_physically_realizable

    # numpy
    numpy = map(lambda m: sympy.matrix2numpy(m, dtype=float), [a_mat, b_mat, c_mat, d_mat])
    numpy_prime = map(lambda m: sympy.matrix2numpy(m, dtype=float),
                      [a_mat_prime, b_mat_prime, c_mat_prime, d_mat_prime])

    assert not System(*numpy).is_physically_realizable
    assert System(*numpy_prime).is_physically_realizable

    # TODO: test with symbolic variables
    # testing with tuned cavity
    gamma = sympy.symbols('gamma', real=True, positive=True)
    eye = sympy.eye(2)

    assert not System(-gamma*eye, eye, -2*gamma*eye, eye).is_physically_realizable
    assert System(-gamma*eye, sympy.sqrt(2*gamma)*eye, -sympy.sqrt(2*gamma)*eye, eye).is_physically_realizable
