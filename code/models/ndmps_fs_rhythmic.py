import numpy as np

import nengo
import nengo.utils.function_space

from . import forcing_functions
from . import oscillator
from . import point_attractor

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace


def generate(data_folder, net=None, alpha=1000.0):
    beta = alpha / 4.0

    # generate the Function Space
    trajectories, _, _ = forcing_functions.load_folder(
        data_folder, rhythmic=True, alpha=alpha, beta=beta)
    # make an array out of all the possible functions we want to represent
    force_space = np.vstack(trajectories)
    # use this array as our space to perform svd over
    fs = nengo.FunctionSpace(space=force_space, n_basis=10)

    # store the weights for each trajectory
    weights_x = []
    weights_y = []
    for ii in range(len(trajectories)):
        forces = force_space[ii*2:ii*2+2]
        # load up the forces to be output by the forcing function
        # calculate the corresponding weights over the basis functions
        weights_x.append(np.dot(fs.basis.T, forces[0]))
        weights_y.append(np.dot(fs.basis.T, forces[1]))

    if net is None:
        net = nengo.Network()
    with net:

        # --------------------- Inputs --------------------------
        net.input = nengo.Node(size_in=2, size_out=2, label='input')

        number = nengo.Node(output=[0], label='number')

        # ------------------- Point Attractors --------------------
        net.x = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)
        nengo.Connection(net.input[0], net.x.input[0], synapse=None)
        net.y = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)
        nengo.Connection(net.input[1], net.y.input[0], synapse=None)

        # -------------------- Oscillators ----------------------
        kick = nengo.Node(nengo.utils.functions.piecewise({0: 1, .05: 0}),
                          label='kick')
        osc = oscillator.generate(net, n_neurons=2000, speed=.025)
        osc.label = 'oscillator'
        nengo.Connection(kick, osc[0])

        # ------------------- Forcing Functions --------------------

        def dmp_weights_func(t, x):
            x = int(min(max(x, 0), len(trajectories)))
            # load weights for generating this number's x and y forces
            return np.hstack([weights_x[x], weights_y[x]])

        # create input switch for generating weights for different numbers
        # NOTE: this should be switched to an associative memory
        dmp_weights_gen = nengo.Node(output=dmp_weights_func,
                                     size_in=1,
                                     size_out=fs.n_basis * 2,
                                     label='dmp weights gen')
        nengo.Connection(number, dmp_weights_gen)

        # -------------------- Product for decoding -----------------------

        product_x = nengo.Network('Product X')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=product_x,
                               input_magnitude=1.0)
        product_y = nengo.Network('Product Y')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=product_y,
                               input_magnitude=1.0)

        # get the largest basis function value for normalization
        max_basis = np.max(fs.basis*fs.scale)
        domain = np.linspace(-np.pi, np.pi, fs.basis.shape[0])
        domain_cossin = np.array([np.cos(domain), np.sin(domain)]).T
        for ff, product in zip([dmp_weights_gen[:fs.n_basis],
                                dmp_weights_gen[fs.n_basis:]],
                               [product_x, product_y]):
            for ii in range(fs.n_basis):
                # find the value of a basis function at a value of (x, y)
                target_function = nengo.utils.connection.target_function(
                    domain_cossin, fs.basis[:, ii]*fs.scale/max_basis)
                nengo.Connection(osc, product.B[ii], **target_function)
                # multiply the value of each basis function at x by its weight
            nengo.Connection(ff, product.A)

        nengo.Connection(product_x.output, net.x.input[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis,
                         synapse=None)
        nengo.Connection(product_y.output, net.y.input[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis,
                         synapse=None)

        # -------------------- Output ------------------------------

        net.output = nengo.Node(size_in=2, size_out=2, label='output')
        nengo.Connection(net.x.output, net.output[0])
        nengo.Connection(net.y.output, net.output[1])

        # create a node to give a plot of the represented function
        ff_plot = fs.make_plot_node(domain=domain, lines=2,
                                    ylim=[-50, 50])
        nengo.Connection(dmp_weights_gen[:fs.n_basis],
                         ff_plot[:fs.n_basis], synapse=0.1)
        nengo.Connection(dmp_weights_gen[fs.n_basis:],
                         ff_plot[fs.n_basis:], synapse=0.1)

    return net
