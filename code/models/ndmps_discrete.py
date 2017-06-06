import numpy as np
from scipy import interpolate

import nengo
import nengo.utils.function_space

import importlib
from . import point_attractor
importlib.reload(point_attractor)
from . import forcing_functions


def generate(data_file, net=None, alpha=1000.0):
    beta = alpha / 4.0

    # generate the forcing function
    y_des = np.load(data_file)['arr_0'].T
    forces, _, goals = forcing_functions.generate(
        y_des=y_des, rhythmic=False, alpha=alpha, beta=beta)

    if net is None:
        net = nengo.Network(label='Discrete NDMP')
    with net:
        # create a start / stop movement signal
        time_func = lambda t: min(max((t * 2) % 4.5 - 2.5, -1), 1)

        # ------------------- Point Attractors ----------------------

        def goal_func(t):
            t = time_func(t)
            if t <= -1:
                return goals[0]
            return goals[1]
        net.goal = nengo.Node(output=goal_func, label='goal')

        net.x = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)
        nengo.Connection(net.goal[0], net.x.input[0], synapse=None)
        net.y = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)
        nengo.Connection(net.goal[1], net.y.input[0], synapse=None)

        # -------------------- Ramp ---------------------------------
        ramp_node = nengo.Node(output=time_func, label='ramp')
        net.ramp = nengo.Ensemble(n_neurons=1000, dimensions=1,
                              label='ramp ens')
        nengo.Connection(ramp_node, net.ramp)

        # ------------------- Forcing Functions ---------------------

        def relay_func(t, x):
            t = time_func(t)
            if t <= -1:
                return [0, 0]
            return x
        # the relay prevents forces from being sent when resetting
        relay = nengo.Node(output=relay_func, size_in=2, size_out=2,
                           label='relay gate')

        domain = np.linspace(-1, 1, len(forces[0]))
        x_func = interpolate.interp1d(domain, forces[0])
        y_func = interpolate.interp1d(domain, forces[1])
        nengo.Connection(net.ramp, relay[0], function=x_func)
        nengo.Connection(net.ramp, relay[1], function=y_func)
        nengo.Connection(relay[0], net.x.input[1], synapse=None)
        nengo.Connection(relay[1], net.y.input[1], synapse=None)

        # -------------------- Output -------------------------------

        net.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(net.x.output, net.output[0], synapse=0.01)
        nengo.Connection(net.y.output, net.output[1], synapse=0.01)

    return net
