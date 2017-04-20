import numpy as np

import nengo

import forcing_functions
import oscillator
import point_attractor


def generate(data_file, net=None):

    # generate our forcing function
    y_des = np.load(data_file)['arr_0'].T
    _, force_functions, _ = forcing_functions.generate(y_des, rhythmic=True)

    if net is None:
        net = nengo.Network(label='Rhythmic NDMP')
    with net:
        # --------------------- Inputs ------------------------------
        net.input = nengo.Node(size_in=2, size_out=2)

        # ------------------- Point Attractors ----------------------
        x = point_attractor.generate(goal=net.input[0], n_neurons=500)
        y = point_attractor.generate(goal=net.input[1], n_neurons=500)

        # -------------------- Oscillators --------------------------
        kick = nengo.Node(
            nengo.utils.functions.piecewise({0: 1, .05: 0}),
            label='kick')
        osc = oscillator.generate(net, n_neurons=3000, speed=.005)
        osc.label = 'oscillator'
        nengo.Connection(kick, osc[0])

        # connect oscillator to point attractor
        nengo.Connection(osc, x.input, synapse=None, **force_functions[0])
        nengo.Connection(osc, y.input, synapse=None, **force_functions[1])

        # -------------------- Output -------------------------------
        net.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(x.output, net.output[0], synapse=None)
        nengo.Connection(y.output, net.output[1], synapse=None)

    return net
