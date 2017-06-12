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

    # generate the Function Space
    forces, _, goals = forcing_functions.load_folder(
        'models/locomotion_trajectories', rhythmic=True,
        alpha=alpha, beta=beta)
    # make an array out of all the possible functions we want to represent
    force_space = np.vstack(forces)
    # use this array as our space to perform svd over
    fs = nengo.FunctionSpace(space=force_space, n_basis=10)

    # store the weights for each movement
    weights_a = []  # ankle
    weights_k = []  # knee
    weights_h = []  # hip
    # NOTE: things are added to weights based on the order files are read
    for ii in range(int(len(goals) / 6)):
        forces = force_space[ii*6:ii*6+6]
        # load up the forces to be output by the forcing function
        # calculate the corresponding weights over the basis functions
        weights_a.append(np.hstack([
            np.dot(fs.basis.T, forces[0]),  # ankle 1
            np.dot(fs.basis.T, forces[1])]))  # ankle 2
        weights_h.append(np.hstack([
            np.dot(fs.basis.T, forces[2]),  # hip 1
            np.dot(fs.basis.T, forces[3])]))  # hip 2
        weights_k.append(np.hstack([
            np.dot(fs.basis.T, forces[4]),  # knee 1
            np.dot(fs.basis.T, forces[5])]))  # knee 2

    # Create our vocabularies
    sps_labels = ['GALLOP', 'RUNNING', 'WALKING']
    rng = np.random.RandomState(0)
    dimensions = 50  # some arbitrary number
    vocab_input = Vocabulary(dimensions=dimensions, rng=rng)
    vocab_dmp_weights_a = Vocabulary(dimensions=fs.n_basis*2, rng=rng)
    vocab_dmp_weights_k = Vocabulary(dimensions=fs.n_basis*2, rng=rng)
    vocab_dmp_weights_h = Vocabulary(dimensions=fs.n_basis*2, rng=rng)

    for ii, (label, wa, wk, wh) in enumerate(zip(
            sps_labels, weights_a, weights_k, weights_h)):
        vocab_input.parse(label)  # randomly generate input vector
        vocab_dmp_weights_a.add(label, wa)
        vocab_dmp_weights_k.add(label, wk)
        vocab_dmp_weights_h.add(label, wh)

    net = spa.SPA()
    net.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with net:

        config = nengo.Config(nengo.Ensemble)
        config[nengo.Ensemble].neuron_type = nengo.Direct()
        with config:
            # --------------------- Inputs --------------------------

            # def input_func(t):
            #     return vocab_input.parse(input_signal).v
            # net.input = nengo.Node(input_func)
            net.input = spa.State(dimensions, subdimensions=10,
                                  vocab=vocab_input)


            # ------------------- Point Attractors --------------------

            zero = nengo.Node([0])
            net.a1 = point_attractor.generate(
                n_neurons=1000, alpha=alpha, beta=beta)
            nengo.Connection(zero, net.a1.input[0], synapse=None)
            net.a2 = point_attractor.generate(
                n_neurons=1000, alpha=alpha, beta=beta)
            nengo.Connection(zero, net.a1.input[0], synapse=None)

            net.k1 = point_attractor.generate(
                n_neurons=1000, alpha=alpha, beta=beta)
            nengo.Connection(zero, net.k1.input[0], synapse=None)
            net.k2 = point_attractor.generate(
                n_neurons=1000, alpha=alpha, beta=beta)
            nengo.Connection(zero, net.k2.input[0], synapse=None)

            net.h1 = point_attractor.generate(
                n_neurons=1000, alpha=alpha, beta=beta)
            nengo.Connection(zero, net.h1.input[0], synapse=None)
            net.h2 = point_attractor.generate(
                n_neurons=1000, alpha=alpha, beta=beta)
            nengo.Connection(zero, net.h2.input[0], synapse=None)

        # -------------------- Oscillators ----------------------

        kick = nengo.Node(nengo.utils.functions.piecewise({0: 1, .05: 0}),
                          label='kick')

        osc = oscillator.generate(net, n_neurons=3000, speed=.01)
        osc.label = 'oscillator'
        nengo.Connection(kick, osc[0])

        # ------------------- Forcing Functions --------------------

        with config:
            net.assoc_mem_a = spa.AssociativeMemory(
                input_vocab=vocab_input,
                output_vocab=vocab_dmp_weights_a,
                wta_output=False)
            nengo.Connection(net.input.output, net.assoc_mem_a.input)

            net.assoc_mem_k = spa.AssociativeMemory(
                input_vocab=vocab_input,
                output_vocab=vocab_dmp_weights_k,
                wta_output=False)
            nengo.Connection(net.input.output, net.assoc_mem_k.input)

            net.assoc_mem_h = spa.AssociativeMemory(
                input_vocab=vocab_input,
                output_vocab=vocab_dmp_weights_h,
                wta_output=False)
            nengo.Connection(net.input.output, net.assoc_mem_h.input)

        # -------------------- Product for decoding -----------------------

            product_a1 = nengo.Network('Product A1')
            nengo.networks.Product(
                n_neurons=1000, dimensions=fs.n_basis, net=product_a1)
            product_a2 = nengo.Network('Product A2')
            nengo.networks.Product(
                n_neurons=1000, dimensions=fs.n_basis, net=product_a2)

            product_h1 = nengo.Network('Product H1')
            nengo.networks.Product(
                n_neurons=1000, dimensions=fs.n_basis, net=product_h1)
            product_h2 = nengo.Network('Product H2')
            nengo.networks.Product(
                n_neurons=1000, dimensions=fs.n_basis, net=product_h2)

            product_k1 = nengo.Network('Product K1')
            nengo.networks.Product(
                n_neurons=1000, dimensions=fs.n_basis, net=product_k1)
            product_k2 = nengo.Network('Product K2')
            nengo.networks.Product(
                n_neurons=1000, dimensions=fs.n_basis, net=product_k2)

            # get the largest basis function value for normalization
            max_basis = np.max(fs.basis*fs.scale)
            domain = np.linspace(-np.pi, np.pi, fs.basis.shape[0])
            domain_cossin = np.array([np.cos(domain), np.sin(domain)]).T
            for ff, product in zip(
                    [net.assoc_mem_a.output[:fs.n_basis],
                     net.assoc_mem_a.output[fs.n_basis:],
                     net.assoc_mem_k.output[:fs.n_basis],
                     net.assoc_mem_k.output[fs.n_basis:],
                     net.assoc_mem_h.output[:fs.n_basis],
                     net.assoc_mem_h.output[fs.n_basis:]],
                    [product_a1, product_a2, product_k1,
                     product_k2, product_h1, product_h2]):
                for ii in range(fs.n_basis):
                    # find the value of a basis function at a value of (x, y)
                    target_function = nengo.utils.connection.target_function(
                        domain_cossin, fs.basis[:, ii]*fs.scale/max_basis)
                    nengo.Connection(osc, product.B[ii], **target_function)
                    # multiply the value of each basis function at x by its weight
                nengo.Connection(ff, product.A)

            nengo.Connection(product_a1.output, net.a1.input[1],
                            transform=np.ones((1, fs.n_basis)) * max_basis)
            nengo.Connection(product_a2.output, net.a2.input[1],
                            transform=np.ones((1, fs.n_basis)) * max_basis)

            nengo.Connection(product_k1.output, net.k1.input[1],
                            transform=np.ones((1, fs.n_basis)) * max_basis)
            nengo.Connection(product_k2.output, net.k2.input[1],
                            transform=np.ones((1, fs.n_basis)) * max_basis)

            nengo.Connection(product_h1.output, net.h1.input[1],
                            transform=np.ones((1, fs.n_basis)) * max_basis)
            nengo.Connection(product_h2.output, net.h2.input[1],
                            transform=np.ones((1, fs.n_basis)) * max_basis)

            # -------------------- Output ------------------------------

            net.output = nengo.Node(size_in=6, label='output')
            nengo.Connection(net.a1.output, net.output[0], synapse=0.01)
            nengo.Connection(net.a2.output, net.output[1], synapse=0.01)
            nengo.Connection(net.k1.output, net.output[2], synapse=0.01)
            nengo.Connection(net.k2.output, net.output[3], synapse=0.01)
            nengo.Connection(net.h1.output, net.output[4], synapse=0.01)
            nengo.Connection(net.h2.output, net.output[5], synapse=0.01)

            # add in the goal offsets
            nengo.Connection(net.assoc_mem_a.output[[-2, -1]],
                             net.output[[0, 1]], synapse=None)
            nengo.Connection(net.assoc_mem_k.output[[-2, -1]],
                             net.output[[2, 3]], synapse=None)
            nengo.Connection(net.assoc_mem_h.output[[-2, -1]],
                             net.output[[4, 5]], synapse=None)

            # create a node to give a plot of the represented function
            ff_plot_a = fs.make_plot_node(domain=domain, lines=2,
                                          ylim=[-1000000, 1000000])
            nengo.Connection(net.assoc_mem_a.output, ff_plot_a, synapse=0.1)

            ff_plot_k = fs.make_plot_node(domain=domain, lines=2,
                                          ylim=[-1000000, 1000000])
            nengo.Connection(net.assoc_mem_k.output, ff_plot_k, synapse=0.1)

            ff_plot_h = fs.make_plot_node(domain=domain, lines=2,
                                          ylim=[-1000000, 1000000])
            nengo.Connection(net.assoc_mem_h.output, ff_plot_h, synapse=0.1)

    return net
