import numpy as np

from inputs import make_cosine_input
from system import System


def test_using_cosine_input():
    sys = System(-1, 1, 1, 1)
    sys.set_input(make_cosine_input(100, 1))

    time_series = np.linspace(0, 1, num=1000)
    dt = time_series[1] - time_series[0]

    for _ in time_series:
        sys.step_evolution(dt)
