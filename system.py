
import numpy as np
import sympy
from numbers import Number
import utils
from inputs import get_wiener_increment


class DimensionError(Exception):
    """
    This class is used to indicate that an error occurred relating to incorrect matrix dimensions
    """
    def __init__(self, message, given=None, expected=None):
        if given is not None and expected is not None:
            message += f" given {given} expected {expected}"
        super(DimensionError, self).__init__(message)

        self.given = given
        self.expected = expected


def _is_expected_system_type(a):
    return isinstance(a, np.ndarray) or isinstance(a, Number) \
           or isinstance(a, sympy.Expr) or isinstance(a, sympy.MatrixBase)


class System:
    """
    This class is for representing state-space models, either numerical (via numpy) or symbolic (via sympy)

    For numerical systems arbitrary inputs can be attached and the system can be simulated in the time-domain,
    but this is not allowed for symbolic systems.

    In abcd symbolic systems are used primarily for finding the physical realizations of systems.
    """
    def __init__(self, a, b, c, d=None):
        """
        Create a new state-space model.

        A TypeError will be thrown if the parameters are anything except numbers, `np.ndarray`s,
        sympy.Expr, or sympy.MatrixBase

        If any sympy elements are used, self.is_symbolic will be set, in which case set_input and step_evolution
        will be disabled.

        A DimensionError will be thrown for all issues regarding the shapes of a, b, c, d.

        :param a: the internal system dynamics of the system. Must be square.
        :param b: the input-coupling matrix. Must have same number of rows as a and columns as b.
        :param c: the output-coupling matrix. Must have the same number of columns (/rows) as a and rows as d.
        :param d: the direct-feed matrix. Will default to identity if b and c imply num_inputs = num_outputs.
        """
        if not (_is_expected_system_type(a) and
                _is_expected_system_type(b) and
                _is_expected_system_type(c) and
                (_is_expected_system_type(d) or d is None)):
            raise TypeError("a, b, c, d should be instances of np.array or Number")

        # if any of the a, b, c, d are sympy objects, then set this to a symbolic system
        self.is_symbolic = any(map(lambda x: isinstance(x, sympy.Expr) or isinstance(x, sympy.MatrixBase), [a, b, c, d]))

        # wrap numbers to become 1x1 matrices
        if isinstance(a, Number) or isinstance(a, sympy.Expr):
            a = [[a]]
        if isinstance(b, Number) or isinstance(b, sympy.Expr):
            b = [[b]]
        if isinstance(c, Number) or isinstance(c, sympy.Expr):
            c = [[c]]
        if isinstance(d, Number) or isinstance(d, sympy.Expr):
            d = [[d]]

        self.a = sympy.Matrix(a) if self.is_symbolic else np.array(a)
        self.b = sympy.Matrix(b) if self.is_symbolic else np.array(b)
        self.c = sympy.Matrix(c) if self.is_symbolic else np.array(c)

        # if no direct-feed matrix is given, then if number of inputs = number of outputs = n,
        # generate n x n identity, otherwise raise DimensionError
        if d is None:
            if self.num_outputs != self.num_inputs:
                raise DimensionError("If no d matrix is given, need num_inputs = num_outputs "
                                     "i.e. num cols in b = num rows in c")
            d = sympy.Identity(self.num_outputs) if self.is_symbolic else np.identity(self.num_outputs)

        self.d = sympy.Matrix(d) if self.is_symbolic else np.array(d)
        self.state = np.zeros((self.num_dof, 1))

        # set all inputs to (white-noise) Wiener processes if system is not symbolic
        if not self.is_symbolic:
            self.inputs = []
            self.set_input(get_wiener_increment)
        else:
            self.inputs = None

        # set variable to keep track of current time
        self._time = None

        # check the dimensions of all the matrices
        if not utils.is_square(self.a):
            raise DimensionError("a should be a square matrix")

        if not self.c.shape[1] == self.num_dof:
            raise DimensionError("a and c need the same number of columns to match number of degrees of freedom",
                                 self.c.shape, (self.num_outputs, self.num_dof))

        if not self.d.shape[0] == self.num_outputs:
            raise DimensionError("c and d need the same number of rows to match number of outputs",
                                 self.d.shape, (self.num_outputs, self.num_inputs))

        if not self.b.shape[0] == self.num_dof:
            raise DimensionError("a and b need the same number of rows to match number of degrees of freedom",
                                 self.b.shape, (self.num_dof, self.num_inputs))

        if not self.d.shape[1] == self.num_inputs:
            raise DimensionError("b and d need the same number of columns to match the number of inputs",
                                 self.d.shape, (self.num_dof, self.num_inputs))

    @property
    def num_dof(self):
        """
        The number of degrees of freedom of the system, given by the number of rows (= num columns) of the a matrix.
        For quantum systems the number of degrees of freedom is effectively half this as we consider each pair of
        internal degrees of freedom to be quadrature or sideband pairs.
        :return: the number of rows of the a matrix
        """
        return self.a.shape[0]

    @property
    def num_inputs(self):
        """
        The number of inputs of the system, given by the number of columns of the b matrix.
        For quantum systems, see the caveat at System.num_dof.__doc__
        :return: the number of columns of the b matrix
        """
        return self.b.shape[1]

    @property
    def num_outputs(self):
        """
        The number of inputs of the system, given by the number of rows of the c matrix.
        For quantum systems, see the caveat at System.num_dof.__doc__
        :return: the number of rows of the c matrix
        """
        return self.c.shape[0]

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        """
        Set system state to specified value, which must have shape (self.num_dof, 1) or be a Number.
        If it is a Number then it sets every degree of freedom to that value
        :param value: the new system state
        """
        if isinstance(value, Number):
            value = np.array([[value] * self.num_dof]).T  # extend the value to a (2, 1) ndarray

        if not isinstance(value, np.ndarray):
            raise TypeError("value must be an np.ndarray or a Number")

        expected_shape = (self.num_dof, 1)
        if value.shape != expected_shape:
            raise DimensionError("value did not have the correct shape", value.shape, expected_shape)
        self._state = value

    @property
    def time_elapsed(self):
        return self._time

    def reset_time(self):
        self._time = None

    def set_input(self, new_input_func):
        """
        For _numerical_ systems, set all inputs to a new input function which takes one argument, the timestep dt
        If system is symbolic, raises TypeError.
        :param new_input_func: input function to replace all inputs with
        """
        if self.is_symbolic:
            raise TypeError("Systems with symbolic elements cannot undergo simulation via time evolution")

        self.inputs = [new_input_func for _ in range(self.num_inputs)]

    def step_evolution(self, dt, t=0):
        """
        For _numerical_ systems, perform state evolution for timestep of dt and return output increments.
        If system is symbolic, raises TypeError.
        :param dt: time step of the evolution
        :param t: initial time
        :return: the increment of the output for this step
        """

        if self.is_symbolic:
            raise TypeError("Systems with symbolic elements cannot undergo simulation via time evolution")

        if self._time is None:
            self._time = t

        input_vector = np.array([[input_func(dt, self._time)] for input_func in self.inputs])

        # increment timestep
        self._time += dt

        # state evolution
        dstate = np.matmul(self.a, self.state) * dt + np.matmul(self.b, input_vector)

        self.state = self.state + dstate

        # input-output evolution
        return np.matmul(self.c, self.state) * dt + np.matmul(self.d, input_vector)

    @property
    def j_matrix(self):
        """
        Get the J matrix ("commutation matrix") for the system assuming the degrees of freedom are (self.num_dof / 2)
        quantum sideband operator pairs

        If self.num_dof is not a multiple of two, a DimensionError will be raised.

        :return: The J matrix for the system, used in the physical realizability condition
        """

        if self.num_dof % 2 != 0:
            raise DimensionError("Quantum systems should have 2n internal states (dofs) where n > 0 is integer")

        num_pairs = self.num_dof // 2
        pairs = [1, -1] * num_pairs

        return sympy.diag(*pairs) if self.is_symbolic else np.diag(pairs)

    @property
    def quantum_t_matrix(self):
        """
        Get the Tw matrix for the system assuming _all_ inputs are (self.num_inputs / 2) quantum sideband operator pairs
        with the system having no classical inputs.

        For example, for self.num_inputs (= num b matrix columns) = 2, the input will be assumed to be the pair
        of operators (u, u*)

        If self.num_inputs is not a multiple of two, a DimensionError will be raised.

        :return: The Tw matrix for the system, used in the physical realizability condition
        """

        if self.num_inputs % 2 != 0:
            raise DimensionError("Quantum systems should have 2n input operators where n > 0 is integer")

        num_pairs = self.num_inputs // 2
        pairs = [1, -1] * num_pairs

        return sympy.diag(*pairs) if self.is_symbolic else np.diag(pairs)

    @property
    def is_physically_realizable(self):
        """
        Return true if the system satisfies the physically realizability conditions, using the J and Tw matrices
        given by self.j_matrix, and self.quantum_t_matrix respectively. The conditions are as follows,

        $$
        A J + J A^d + B Tw B^d = 0,
        J C^d + B Tw D^d = 0
        $$

        where ^d denotes Hermitian conjugate

        :return: whether or not the state-space is physically realizable
        """

        j = self.j_matrix
        tw = self.quantum_t_matrix

        if self.is_symbolic:
            assert isinstance(self.a, sympy.MatrixBase)

            cond1 = self.a * j + j * self.a.H + self.b * tw * self.b.H == sympy.zeros(*self.a.shape)
            cond2 = j * self.c.H + self.b * tw * self.d.H == sympy.zeros(*(j * self.c.H).shape)

        else:
            assert isinstance(self.a, np.ndarray)

            cond1 = np.array_equal(np.matmul(self.a, j) + np.matmul(j, self.a.conj().T)
                                   + np.matmul(np.matmul(self.b, tw), self.b.conj().T),
                                   np.zeros(np.matmul(self.a, j).shape))

            cond2 = np.array_equal(np.matmul(j, self.c.conj().T) + np.matmul(np.matmul(self.b, tw), self.d.conj().T),
                                   np.zeros(np.matmul(j, self.c.conj().T).shape))

        return cond1 and cond2
