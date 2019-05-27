import sys
import numpy as np
from runner import run_and_plot_1dof_1output_system
from system import System
from inputs import make_coherent_wiener_func, make_cosine_input


def example1():
    gamma = 100
    system = System(-gamma, np.sqrt(2 * gamma), -np.sqrt(2 * gamma), 1)
    system.set_input(make_coherent_wiener_func(10))
    # system.state = 0
    run_and_plot_1dof_1output_system(system)


choices = [example1]

if __name__ == "__main__":
    choice = sys.argv.pop()

    try:
        choice = int(choice)
    except ValueError:
        print("error: invalid number", file=sys.stderr)
        sys.exit(-1)

    if choice < 1 or choice > len(choices) + 1:
        print(f"error: choose from 1 to {len(choices)}", file=sys.stderr)
        sys.exit(-1)

    choices[choice-1]()
