import numpy as np
from scipy import interpolate

import nengo
import nengo.utils.function_space

import point_attractor
import forcing_functions


def generate(data_file, net=None):

    # generate the forcing function
    y_des = np.load(data_file)['arr_0'].T
    forces, _, goals = forcing_functions.generate(y_des=y_des, rhythmic=False)

    if net is None:
        net = nengo.Network(label='Discrete NDMP')
    with net:
        # --------------------- Inputs ------------------------------
        net.input = nengo.Node(size_in=1, size_out=1)

        # create a start / stop movement signal
        time_func = lambda t: min(max((t * 1) % 4.5 - 2.5, -1), 1)

        # ------------------- Point Attractors ----------------------

        def goal_func(t):
            t = time_func(t)
            if t <= -1:
                return goals[0]
            return goals[1]
        net.goal = nengo.Node(output=goal_func, label='goal')

        net.x = point_attractor.generate(net.goal[0], n_neurons=1000)
        net.y = point_attractor.generate(net.goal[1], n_neurons=1000)

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
        nengo.Connection(relay[0], net.x.input)
        nengo.Connection(relay[1], net.y.input)

        # -------------------- Output -------------------------------

        net.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(net.x.output, net.output[0], synapse=.01)
        nengo.Connection(net.y.output, net.output[1], synapse=.01)

    return net
