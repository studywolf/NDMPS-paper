import numpy as np

import nengo
import nengo.utils.function_space
import nengo.spa as spa
from nengo.spa import Vocabulary

import forcing_functions
import oscillator
import point_attractor

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace


def generate(input_signal):

    # generate the Function Space
    forces, _, goals = forcing_functions.load_folder(
        'locomotion_trajectories', rhythmic=True)
    # make an array out of all the possible functions we want to represent
    force_space = np.vstack(forces)
    print(np.array(goals))
    # use this array as our space to perform svd over
    fs = nengo.FunctionSpace(space=force_space, n_basis=20)

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
    vocab_dmp_weights_a = Vocabulary(dimensions=fs.n_basis*2 + 2, rng=rng)
    vocab_dmp_weights_k = Vocabulary(dimensions=fs.n_basis*2 + 2, rng=rng)
    vocab_dmp_weights_h = Vocabulary(dimensions=fs.n_basis*2 + 2, rng=rng)

    for ii, (label, wa, wk, wh) in enumerate(zip(
            sps_labels, weights_a, weights_k, weights_h)):
        vocab_input.parse(label)  # randomly generate input vector

        vocab_dmp_weights_a.add(
            label, np.hstack([wa, goals[ii*6+0], goals[ii*6+1]]))
        vocab_dmp_weights_k.add(
            label, np.hstack([wk, goals[ii*6+4], goals[ii*6+5]]))
        vocab_dmp_weights_h.add(
            label, np.hstack([wh, goals[ii*6+2], goals[ii*6+3]]))

    net = spa.SPA()
    net.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with net:

        config = nengo.Config(nengo.Ensemble)
        config[nengo.Ensemble].neuron_type = nengo.Direct()
        with config:
            # --------------------- Inputs --------------------------
            net.assoc_mem_a = spa.AssociativeMemory(
                input_vocab=vocab_input,
                output_vocab=vocab_dmp_weights_a,
                wta_output=False)
            net.assoc_mem_k = spa.AssociativeMemory(
                input_vocab=vocab_input,
                output_vocab=vocab_dmp_weights_k,
                wta_output=False)
            net.assoc_mem_h = spa.AssociativeMemory(
                input_vocab=vocab_input,
                output_vocab=vocab_dmp_weights_h,
                wta_output=False)

            # def input_func(t):
            #     return vocab_input.parse(input_signal).v
            # net.input = nengo.Node(input_func)
            net.input = spa.State(dimensions, subdimensions=10,
                                  vocab=vocab_input)

            nengo.Connection(net.input.output, net.assoc_mem_a.input)
            nengo.Connection(net.input.output, net.assoc_mem_k.input)
            nengo.Connection(net.input.output, net.assoc_mem_h.input)

            # ------------------- Point Attractors --------------------

            zero = nengo.Node([0])
            net.a1 = point_attractor.generate(zero, n_neurons=1000)
            net.a2 = point_attractor.generate(zero, n_neurons=1000)

            net.k1 = point_attractor.generate(zero, n_neurons=1000)
            net.k2 = point_attractor.generate(zero, n_neurons=1000)

            net.h1 = point_attractor.generate(zero, n_neurons=1000)
            net.h2 = point_attractor.generate(zero, n_neurons=1000)

        # -------------------- Oscillators ----------------------

        kick = nengo.Node(nengo.utils.functions.piecewise({0: 1, .05: 0}),
                          label='kick')

        osc = oscillator.generate(net, n_neurons=3000, speed=.01)
        osc.label = 'oscillator'
        nengo.Connection(kick, osc[0])

        # ------------------- Forcing Functions --------------------

        # n_basis_functions dimensions to represent the weights, + 1 to
        # represent the x position to decode from
        with config:
            # TODO: are these needed what's the deal?
            ff_a1 = nengo.Ensemble(n_neurons=1000, dimensions=fs.n_basis,
                                   radius=np.sqrt(fs.n_basis), label='ff a')
            ff_a2 = nengo.Ensemble(n_neurons=1000, dimensions=fs.n_basis,
                                   radius=np.sqrt(fs.n_basis), label='ff a')

            ff_k1 = nengo.Ensemble(n_neurons=1000, dimensions=fs.n_basis,
                                   radius=np.sqrt(fs.n_basis), label='ff k')
            ff_k2 = nengo.Ensemble(n_neurons=1000, dimensions=fs.n_basis,
                                   radius=np.sqrt(fs.n_basis), label='ff k')

            ff_h1 = nengo.Ensemble(n_neurons=1000, dimensions=fs.n_basis,
                                   radius=np.sqrt(fs.n_basis), label='ff h')
            ff_h2 = nengo.Ensemble(n_neurons=1000, dimensions=fs.n_basis,
                                   radius=np.sqrt(fs.n_basis), label='ff h')
        # hook up input
        nengo.Connection(
            net.assoc_mem_a.output[:fs.n_basis], ff_a1, synapse=.01)
        nengo.Connection(
            net.assoc_mem_a.output[fs.n_basis:2*fs.n_basis], ff_a2, synapse=.01)

        nengo.Connection(
            net.assoc_mem_k.output[:fs.n_basis], ff_k1, synapse=.01)
        nengo.Connection(
            net.assoc_mem_k.output[fs.n_basis:2*fs.n_basis], ff_k2, synapse=.01)

        nengo.Connection(
            net.assoc_mem_h.output[:fs.n_basis], ff_h1, synapse=.01)
        nengo.Connection(
            net.assoc_mem_h.output[fs.n_basis:2*fs.n_basis], ff_h2, synapse=.01)

        # -------------------- Product for decoding -----------------------

        with config:
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
                    [ff_a1, ff_a2, ff_k1, ff_k2, ff_h1, ff_h2],
                    [product_a1, product_a2, product_k1,
                     product_k2, product_h1, product_h2]):
                for ii in range(fs.n_basis):
                    # find the value of a basis function at a value of (x, y)
                    target_function = nengo.utils.connection.target_function(
                        domain_cossin, fs.basis[:, ii]*fs.scale/max_basis)
                    nengo.Connection(osc, product.B[ii], **target_function)
                    # multiply the value of each basis function at x by its weight
                    nengo.Connection(ff[ii], product.A[ii])

            nengo.Connection(product_a1.output, net.a1.input,
                            transform=np.ones((1, fs.n_basis)) * max_basis)
            nengo.Connection(product_a2.output, net.a2.input,
                            transform=np.ones((1, fs.n_basis)) * max_basis)

            nengo.Connection(product_k1.output, net.k1.input,
                            transform=np.ones((1, fs.n_basis)) * max_basis)
            nengo.Connection(product_k2.output, net.k2.input,
                            transform=np.ones((1, fs.n_basis)) * max_basis)

            nengo.Connection(product_h1.output, net.h1.input,
                            transform=np.ones((1, fs.n_basis)) * max_basis)
            nengo.Connection(product_h2.output, net.h2.input,
                            transform=np.ones((1, fs.n_basis)) * max_basis)

            # -------------------- Output ------------------------------

            net.output = nengo.Node(size_in=6, size_out=6, label='output')
            nengo.Connection(net.a1.output, net.output[0], synapse=None)
            nengo.Connection(net.a2.output, net.output[1], synapse=None)
            nengo.Connection(net.k1.output, net.output[2], synapse=None)
            nengo.Connection(net.k2.output, net.output[3], synapse=None)
            nengo.Connection(net.h1.output, net.output[4], synapse=None)
            nengo.Connection(net.h2.output, net.output[5], synapse=None)

            # add in the goal offsets
            nengo.Connection(net.assoc_mem_a.output[[-2, -1]],
                             net.output[[0, 1]], synapse=None)
            nengo.Connection(net.assoc_mem_k.output[[-2, -1]],
                             net.output[[2, 3]], synapse=None)
            nengo.Connection(net.assoc_mem_h.output[[-2, -1]],
                             net.output[[4, 5]], synapse=None)

            # create a node to give a plot of the represented function
            ff_plot_a = fs.make_plot_node(domain=domain, lines=2,
                                        min_y=-5000, max_y=5000)
            nengo.Connection(ff_a1, ff_plot_a[:fs.n_basis], synapse=0.1)
            nengo.Connection(ff_a2, ff_plot_a[fs.n_basis:], synapse=0.1)

            ff_plot_k = fs.make_plot_node(domain=domain, lines=2,
                                        min_y=-5000, max_y=5000)
            nengo.Connection(ff_k1, ff_plot_k[:fs.n_basis], synapse=0.1)
            nengo.Connection(ff_k2, ff_plot_k[fs.n_basis:], synapse=0.1)

            ff_plot_h = fs.make_plot_node(domain=domain, lines=2,
                                        min_y=-5000, max_y=5000)
            nengo.Connection(ff_h1, ff_plot_h[:fs.n_basis], synapse=0.1)
            nengo.Connection(ff_h2, ff_plot_h[fs.n_basis:], synapse=0.1)

    return net
