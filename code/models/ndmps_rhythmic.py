import numpy as np

import nengo

from . import forcing_functions
from . import oscillator
from . import point_attractor


def generate(data_file, net=None, alpha=1000.0):
    beta = alpha / 4.0

    # generate our forcing function
    y_des = np.load(data_file)['arr_0'].T
    _, force_functions, _ = forcing_functions.generate(
        y_des, rhythmic=True, alpha=alpha, beta=beta)

    if net is None:
        net = nengo.Network(label='Rhythmic NDMP')
    with net:
        # --------------------- Inputs ------------------------------
        net.input = nengo.Node(size_in=2, size_out=2)

        # ------------------- Point Attractors ----------------------
        x = point_attractor.generate(
            n_neurons=500, alpha=alpha, beta=beta)
        nengo.Connection(net.input[0], x.input[0], synapse=None)
        y = point_attractor.generate(
            n_neurons=500, alpha=alpha, beta=beta)
        nengo.Connection(net.input[1], y.input[0], synapse=None)

        # -------------------- Oscillators --------------------------
        kick = nengo.Node(
            nengo.utils.functions.piecewise({0: 1, .05: 0}),
            label='kick')
        osc = oscillator.generate(net, n_neurons=3000, speed=.025)
        osc.label = 'oscillator'
        nengo.Connection(kick, osc[0])

        # connect oscillator to point attractor
        nengo.Connection(osc, x.input[1], synapse=None, **force_functions[0])
        nengo.Connection(osc, y.input[1], synapse=None, **force_functions[1])

        # -------------------- Output -------------------------------
        net.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(x.output, net.output[0], synapse=None)
        nengo.Connection(y.output, net.output[1], synapse=None)

    return net
