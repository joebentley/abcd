
import numpy as np
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


class System:
    def __init__(self, a, b, c, d=None):
        """
        Create a new state-space model.

        A TypeError will be thrown if the parameters are anything except numbers or `np.ndarray`s.

        A DimensionError will be thrown for all issues regarding the shapes of a, b, c, d.

        :param a: the internal system dynamics of the system. Must be square.
        :param b: the input-coupling matrix. Must have same number of rows as a and columns as b.
        :param c: the output-coupling matrix. Must have the same number of columns (/rows) as a and rows as d.
        :param d: the direct-feed matrix. Will default to identity if b and c imply num_inputs = num_outputs.
        """
        if not ((isinstance(a, np.ndarray) or isinstance(a, Number)) and
                (isinstance(b, np.ndarray) or isinstance(b, Number)) and
                (isinstance(c, np.ndarray) or isinstance(c, Number)) and
                (isinstance(d, np.ndarray) or isinstance(d, Number) or d is None)):
            raise TypeError("a, b, c, d should be instances of np.array or Number")

        # wrap numbers to become 1x1 matrices
        if isinstance(a, Number):
            a = [[a]]
        if isinstance(b, Number):
            b = [[b]]
        if isinstance(c, Number):
            c = [[c]]
        if isinstance(d, Number):
            d = [[d]]

        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

        # if no direct-feed matrix is given, then if number of inputs = number of outputs = n,
        # generate n x n identity, otherwise raise DimensionError
        if d is None:
            if self.num_outputs != self.num_inputs:
                raise DimensionError("If no d matrix is given, need num_inputs = num_outputs"
                                     "i.e. num cols in b = num rows in c")
            d = np.identity(self.num_outputs)

        self.d = np.array(d)
        self.state = np.zeros((self.num_dof, 1))

        # set all inputs to (white-noise) Wiener processes
        self.inputs = []
        self.set_input(get_wiener_increment)

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
        return self.a.shape[0]

    @property
    def num_inputs(self):
        return self.b.shape[1]

    @property
    def num_outputs(self):
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
        Set all inputs to a new input function which takes one argument, the timestep dt
        :param new_input_func: input function to replace all inputs with
        """
        self.inputs = [new_input_func for _ in range(self.num_inputs)]

    def step_evolution(self, dt, t=0):
        """
        Perform state evolution for timestep of dt and return output increments
        :param dt: time step of the evolution
        :param t: initial time
        :return: the increment of the output for this step
        """

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
