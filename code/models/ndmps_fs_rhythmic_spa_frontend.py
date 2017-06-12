import numpy as np

import nengo
import nengo.utils.function_space
import nengo.spa as spa
from nengo.spa import Vocabulary

from . import forcing_functions
from . import oscillator
from . import point_attractor

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace


def generate(input_signal, alpha=1000.0):
    beta = alpha / 4.0

    # Read in the class mean for numbers from vision network
    weights_data = np.load('models/mnist_vision_data/params.npz')
    weights = weights_data['Wc']
    means_data = np.load('models/mnist_vision_data/class_means.npz')
    means = np.matrix(1.0 / means_data['means'])
    sps = np.multiply(weights.T, means.T)[:10]
    sps_labels = [
        'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR',
        'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
    dimensions = weights.shape[0]

    # generate the Function Space
    forces, _, _ = forcing_functions.load_folder(
        'models/handwriting_trajectories', rhythmic=True,
        alpha=alpha, beta=beta)
    # make an array out of all the possible functions we want to represent
    force_space = np.vstack(forces)
    # use this array as our space to perform svd over
    fs = nengo.FunctionSpace(space=force_space, n_basis=10)

    # store the weights for each trajectory
    weights_x = []
    weights_y = []
    for ii in range(len(forces)):
        forces = force_space[ii*2:ii*2+2]
        # load up the forces to be output by the forcing function
        # calculate the corresponding weights over the basis functions
        weights_x.append(np.dot(fs.basis.T, forces[0]))
        weights_y.append(np.dot(fs.basis.T, forces[1]))

    # Create our vocabularies
    rng = np.random.RandomState(0)
    vocab_vision = Vocabulary(dimensions=dimensions, rng=rng)
    vocab_dmp_weights_x = Vocabulary(dimensions=fs.n_basis, rng=rng)
    vocab_dmp_weights_y = Vocabulary(dimensions=fs.n_basis, rng=rng)
    for label, sp, wx, wy in zip(
            sps_labels, sps, weights_x, weights_y):
        vocab_vision.add(
            label, np.array(sp)[0] / np.linalg.norm(np.array(sp)[0]))
        vocab_dmp_weights_x.add(label, wx)
        vocab_dmp_weights_y.add(label, wy)


    net = spa.SPA()
    # net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:

        # --------------------- Inputs --------------------------
        # def input_func(t):
        #     return vocab_vision.parse(input_signal).v
        # net.input = nengo.Node(input_func, label='input')
        net.input = spa.State(dimensions, subdimensions=10,
                              vocab=vocab_vision)

        number = nengo.Node(output=[0], label='number')

        # ------------------- Point Attractors --------------------
        net.x = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)
        net.y = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)

        # -------------------- Oscillators ----------------------
        kick = nengo.Node(nengo.utils.functions.piecewise({0: 1, .05: 0}),
                          label='kick')
        osc = oscillator.generate(net, n_neurons=2000, speed=.05)
        osc.label = 'oscillator'
        nengo.Connection(kick, osc[0])

        # ------------------- Forcing Functions --------------------

        net.assoc_mem_x = spa.AssociativeMemory(
            input_vocab=vocab_vision,
            output_vocab=vocab_dmp_weights_x,
            wta_output=False)
        nengo.Connection(net.input.output, net.assoc_mem_x.input)

        net.assoc_mem_y = spa.AssociativeMemory(
            input_vocab=vocab_vision,
            output_vocab=vocab_dmp_weights_y,
            wta_output=False)
        nengo.Connection(net.input.output, net.assoc_mem_y.input)

        # -------------------- Product for decoding -----------------------

        net.product_x = nengo.Network('Product X')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=net.product_x,
                               input_magnitude=1.0)
        net.product_y = nengo.Network('Product Y')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=net.product_y,
                               input_magnitude=1.0)

        # get the largest basis function value for normalization
        max_basis = np.max(fs.basis*fs.scale)
        domain = np.linspace(-np.pi, np.pi, fs.basis.shape[0])
        domain_cossin = np.array([np.cos(domain), np.sin(domain)]).T
        for ff, product in zip([net.assoc_mem_x.output,
                                net.assoc_mem_y.output],
                               [net.product_x, net.product_y]):
            for ii in range(fs.n_basis):
                # find the value of a basis function at a value of (x, y)
                target_function = nengo.utils.connection.target_function(
                    domain_cossin, fs.basis[:, ii]*fs.scale/max_basis)
                nengo.Connection(osc, product.B[ii], **target_function)
                # multiply the value of each basis function at x by its weight
            nengo.Connection(ff, product.A)

        nengo.Connection(net.product_x.output, net.x.input[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis,
                         synapse=None)
        nengo.Connection(net.product_y.output, net.y.input[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis,
                         synapse=None)

        # -------------------- Output ------------------------------

        net.output = nengo.Node(size_in=2, label='output')
        nengo.Connection(net.x.output, net.output[0])
        nengo.Connection(net.y.output, net.output[1])

        # create a node to give a plot of the represented function
        ff_plot = fs.make_plot_node(domain=domain, lines=2,
                                    ylim=[-1000000, 1000000])
        nengo.Connection(net.assoc_mem_x.output,
                         ff_plot[:fs.n_basis], synapse=0.1)
        nengo.Connection(net.assoc_mem_y.output,
                         ff_plot[fs.n_basis:], synapse=0.1)

    return net
