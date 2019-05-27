from scipy.stats import norm
import numpy as np


def get_wiener_increment(dt, t):
    """
    Get the Wiener (white-noise) increment dW for timestep dt

    dW is sampled from Gaussian with mean of 0, variance of dt

    :param dt: the time increment
    :return: a Wiener increment dW
    """
    return norm.rvs(scale=dt)


def get_coherent_wiener(mean, dt):
    return norm.rvs(loc=mean, scale=dt)


def make_coherent_wiener_func(mean):
    """
    Return a "coherent" (non-zero mean) wiener process with given mean
    :param mean: the centre of the wiener process
    :return: a Wiener process with offset mean, taking one argument dt and returning the Wiener increment
    """
    return lambda dt, t: get_coherent_wiener(mean, dt)


def get_cosine_increment(frequency, amplitude, dt, t):
    # u(t) = A sin(2 pi f t)
    # du(t) = 2 pi f A cos(2 pi f t) dt
    return 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t) * dt


def make_cosine_input(frequency, amplitude=1):
    return lambda dt, t: get_cosine_increment(frequency, amplitude, dt, t)
