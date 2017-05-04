import numpy as np
from scipy import interpolate

import nengo
import nengo.utils.function_space

import point_attractor
import forcing_functions

def angle(x):
   return np.arctan2(x[3]-x[1], x[2]-x[0])

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def generate(data_file):

    # generate the forcing function
    y_des = np.load(data_file)['arr_0'].T
    forces, _, goals = forcing_functions.generate(
        y_des=y_des, rhythmic=False, generalized=True)

    net = nengo.Network(label='Discrete NDMP')
    # net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:
        # --------------------- Inputs ------------------------------
        net.input = nengo.Node(size_in=1, size_out=1)

        # create a start / stop movement signal
        time_func = lambda t: min(max((t * .5) % 4.5 - 2.5, -1), 1)

        # ------------------- Point Attractors ----------------------

        def goal_relay_func(t, x):
            return np.hstack([x, angle(x)])
        net.goal_relay = nengo.Node(goal_relay_func,
                                    size_in=4, size_out=5)

        def goal_transformed(t, x):
            theta = angle(x)
            R = rotation_matrix(theta)
            x = np.array([x[2] - x[0], x[3] - x[1]])
            return np.hstack([0, 0, np.dot(R.T, x)])
        net.goal_transformed = nengo.Node(
            goal_transformed, size_in=4, size_out=4)

        def goal_func(t, x):
            t = time_func(t)
            if t <= -1:
                return x[:2]
            return x[2:]
        net.goal = nengo.Node(output=goal_func,
                              size_in=4, size_out=2,
                              label='goal')
        nengo.Connection(net.goal_transformed, net.goal, synapse=None)

        net.x = point_attractor.generate(net.goal[0], n_neurons=1000)
        net.y = point_attractor.generate(net.goal[1], n_neurons=1000)

        # -------------------- Ramp ---------------------------------
        ramp_node = nengo.Node(output=time_func, label='ramp')
        ramp = nengo.Ensemble(n_neurons=4000, dimensions=2,
                               radius=3, label='ramp ens 1',)
        nengo.Connection(ramp_node, ramp[0])

        # ------------------- Forcing Functions ---------------------

        def relay_func(t, x):
            t = time_func(t)
            if t <= -1:
                return [0, 0]
            return x
        # the relay prevents forces from being sent when resetting
        relay = nengo.Node(output=relay_func, size_in=2, size_out=2,
                           label='relay gate')

        # create the forcing functions
        domain = np.linspace(-1, 1, len(forces[0]))
        x_func = interpolate.interp1d(domain, forces[0])
        y_func = interpolate.interp1d(domain, forces[1])
        def calc_ff(x):
            # clip x so that it's inside the interpolation function range
            gain = np.copy(x[1])
            x = min(max(x[0], -1), 1)
            # add gain term to scale the movement
            xy = np.array([x_func(x), y_func(x)]) * gain
            return xy
        nengo.Connection(net.goal_transformed[2], ramp[1])
        nengo.Connection(ramp, relay[:2], function=calc_ff)
        nengo.Connection(relay[0], net.x.input)
        nengo.Connection(relay[1], net.y.input)

        # -------------------- Output -------------------------------

        def rotate(x):
            R = rotation_matrix(x[0])
            return np.dot(R, np.array([x[1], x[2]]))
        net.rotate = nengo.Ensemble(n_neurons=3000, dimensions=3, radius=3)
        nengo.Connection(net.x.output, net.rotate[1], synapse=.01)
        nengo.Connection(net.y.output, net.rotate[2], synapse=.01)
        nengo.Connection(net.goal_relay[4], net.rotate[0])

        net.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(net.rotate, net.output, function=rotate)

    return net, goals
