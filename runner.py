import numpy as np
import matplotlib.pyplot as plt
from system import System
from inputs import make_cosine_input


def run_until_steady_state(system: System, max_iterations=1000, rel_criterion=1e-4, dt=0.001, return_iterations=False):
    """
    Run the system until the maximum relative change between two state increments is less than rel_criterion.
    If it does not settle before the number if iterations reaches max_iterations, it will return None.
    :param system: the state-space system to run
    :param max_iterations: is the maximum number of iterations before giving up
    :param rel_criterion: is the relative change required for steady state,
    :param dt: is the timestep used for state evolution
    :param return_iterations: is a boolean determining whether to return a tuple with the
    number of iterations taken as second argument
    :return:
    """

    # initial state
    state = system.state

    # only try up to max_iterations
    for i in range(max_iterations):
        previous_state = state  # set previous state to current state
        state = system.step_evolution(dt)  # set current state to next state

        # check if maximum fractional change between current and previous state is within our tolerance
        if max(abs(state - previous_state))/max(abs(state)) < rel_criterion:
            return (state, i) if return_iterations else state

    return None


def measure_transfer_function(system: System):
    system.state = np.zeros((system.num_dof, 1))

    pass


def run_and_collect_results(system: System, time_array: np.ndarray):
    """
    Run the system and collect the states, and the output increments cumulatively
    :param system: the state-space system to evaluate
    :param time_array: the time series for which to generate the data
    :return: a tuple (states, outputs) where
    states is a (len(time_array), system.num_dof) np.ndarray instance representing the system state over time and
    outputs is a (len(time_array), system.num_outputs) np.ndarray instance representing the system output over time,
    added up cumulatively from the output increments return by system.step_evolution
    """
    if not isinstance(time_array, np.ndarray):
        raise TypeError("time_array needs to be an ndarray")

    states = np.zeros((system.num_dof, len(time_array)))
    outputs = np.zeros((system.num_outputs, len(time_array)))
    dt = time_array[1] - time_array[0]

    states[:, 0] = system.state.reshape(1, system.num_dof)

    for i in range(len(time_array) - 1):
        outputs[:, i+1] = outputs[:, i] + system.step_evolution(dt).reshape(1, system.num_outputs)
        states[:, i+1] = system.state.reshape(1, system.num_dof)

    return states, outputs


def run_and_plot_1dof_1output_system(system: System, until_time=1, dt=0.001):
    if system.num_dof != 1 and system.num_outputs != 1:
        raise ValueError("System must have 1 degree of freedom and 1 output")

    time_series = np.linspace(0, until_time, num=int(until_time/dt))

    (states, outputs) = run_and_collect_results(system, time_series)

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(time_series, states.T)
    axes[1].plot(time_series, outputs.T)

    plt.show()
