import numpy as np

from system import System
from runner import run_until_steady_state
from inputs import make_coherent_wiener_func


def test_stable_steady_state_settles():
    sys = System(-1, 1, 1, 1)  # this system is stable
    sys.set_input(make_coherent_wiener_func(10))

    steady_state = run_until_steady_state(sys)
    assert steady_state is not None and isinstance(steady_state, np.ndarray)

    # getting number of iterations
    _, iterations = run_until_steady_state(sys, return_iterations=True)
    assert iterations is not None and iterations > 0


def test_unstable_steady_state_does_not_settle():
    sys = System(1, 1, 1, 1)  # this system is _unstable_
    sys.set_input(make_coherent_wiener_func(10))

    steady_state = run_until_steady_state(sys)
    assert steady_state is None
