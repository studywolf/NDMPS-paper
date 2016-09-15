import numpy as np
from scipy import interpolate

import nengo


def gen_forcing_functions(y_des, dt=.001, num_samples=1000,
                          alpha=10, beta=10/4., rhythmic=False):

    # scale our trajectory and find the center point
    y_des = y_des.T
    if rhythmic is True:
        goal = np.sum(y_des, axis=1) / y_des.shape[1]
    else:
        goal = np.copy(y_des[:, -1])

    # interpolate our desired trajectory to smooth out the sampling
    path = np.zeros((y_des.shape[0], num_samples))
    x = np.linspace(-1, 1, y_des.shape[1])
    for d in range(y_des.shape[0]):
        path_gen = interpolate.interp1d(x, y_des[d])
        for ii, t in enumerate(np.linspace(-1, 1, num_samples)):
            path[d, ii] = path_gen(t)
    y_des = path

    # calculate velocity of y_des
    dy_des = np.diff(y_des) / (dt * num_samples)
    # add zero to the beginning of every row
    dy_des = np.hstack((np.zeros((y_des.shape[0], 1)), dy_des))

    # calculate acceleration of y_des
    ddy_des = np.diff(dy_des) / (dt * num_samples)
    # add zero to the beginning of every row
    ddy_des = np.hstack((np.zeros((y_des.shape[0], 1)), ddy_des))

    forces = []
    forcing_functions = []
    x = np.linspace(-np.pi, np.pi, num_samples)
    for ii in range(y_des.shape[0]):
        # find the force required to move along this trajectory
        # by subtracting out the effects of the point attractor
        forces.append(ddy_des[ii] - alpha *
                      (beta * (goal[ii] - y_des[ii]) - dy_des[ii]))
        # create the lookup table set for training oscillator output
        forcing_functions.append(
            nengo.utils.connection.target_function(
                np.array([np.sin(x), np.cos(x)]).T,
                forces[ii]))

    return forces, forcing_functions
