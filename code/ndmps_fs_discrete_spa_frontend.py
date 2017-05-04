import numpy as np

import nengo
import nengo.utils.function_space
import nengo.spa as spa
from nengo.spa import Vocabulary

import forcing_functions
import point_attractor

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace


def generate(input_signal):

    # Read in the class mean for numbers from vision network
    weights_data = np.load('params.npz')
    weights = weights_data['Wc']
    means_data = np.load('class_means.npz')
    means = np.matrix(1.0 / means_data['means'])
    sps = np.multiply(weights.T, means.T)[:10]
    sps_labels = [
        'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR',
        'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
    dimensions = weights.shape[0]

    # generate the Function Space
    forces, _, goals = forcing_functions.load_folder(
        'handwriting_trajectories', rhythmic=False)
    # make an array out of all the possible functions we want to represent
    force_space = np.vstack(forces)
    # use this array as our space to perform svd over
    fs = nengo.FunctionSpace(space=force_space, n_basis=10)

    # store the weights for each number
    weights_x = []
    weights_y = []
    for ii in range(len(goals)):
        forces = force_space[ii*2:ii*2+2]
        # load up the forces to be output by the forcing function
        # calculate the corresponding weights over the basis functions
        weights_x.append(np.dot(fs.basis.T, forces[0]))
        weights_y.append(np.dot(fs.basis.T, forces[1]))


    # Create our vocabularies
    rng = np.random.RandomState(0)
    vocab_vision = Vocabulary(dimensions=dimensions, rng=rng)
    vocab_dmp_weights_x = Vocabulary(dimensions=fs.n_basis + 2, rng=rng)
    vocab_dmp_weights_y = Vocabulary(dimensions=fs.n_basis + 2, rng=rng)
    for label, sp, wx, wy, goal in zip(
            sps_labels, sps, weights_x, weights_y, goals):
        vocab_vision.add(
            label, np.array(sp)[0] / np.linalg.norm(np.array(sp)[0]))
        vocab_dmp_weights_x.add(
            label, np.hstack([wx, goal[0][0], goal[1][0]]))
        vocab_dmp_weights_y.add(
            label, np.hstack([wy, goal[0][1], goal[1][1]]))


    net = spa.SPA()
    # net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:

        net.assoc_mem_x = spa.AssociativeMemory(
            input_vocab=vocab_vision,
            output_vocab=vocab_dmp_weights_x,
            wta_output=False)

        net.assoc_mem_y = spa.AssociativeMemory(
            input_vocab=vocab_vision,
            output_vocab=vocab_dmp_weights_y,
            wta_output=False)

        def input_func(t):
            return vocab_vision.parse(input_signal).v
        net.input = nengo.Node(input_func)
        # net.input = spa.State(dimensions, subdimensions=10,
        #                       vocab=vocab_vision)

        nengo.Connection(net.input, net.assoc_mem_x.input)
        nengo.Connection(net.input, net.assoc_mem_y.input)


        time_func = lambda t: min(max((t * 1) % 4 - 2.5, -1), 1)
        timer_node = nengo.Node(output=time_func)
        # ------------------- Point Attractors --------------------

        def goals_func(t, x):
            if (x[0] + 1) < 1e-5:
                return x[1], x[2]
            return x[3], x[4]
        goal_node = nengo.Node(goals_func, size_in=5, size_out=2)
        nengo.Connection(timer_node, goal_node[0])
        nengo.Connection(net.assoc_mem_x.output[[-2, -1]], goal_node[[1, 3]])
        nengo.Connection(net.assoc_mem_y.output[[-2, -1]], goal_node[[2, 4]])

        net.x = point_attractor.generate(goal_node[0], n_neurons=1000)
        net.y = point_attractor.generate(goal_node[1], n_neurons=1000)

        # -------------------- Ramp ------------------------------
        ramp_node = nengo.Node(output=time_func)
        ramp = nengo.Ensemble(n_neurons=1000, dimensions=1, label='ramp')
        nengo.Connection(ramp_node, ramp)

        # ------------------- Forcing Functions --------------------

        # n_basis_functions dimensions to represent the weights, + 1 to
        # represent the x position to decode from
        ff_x = nengo.Ensemble(n_neurons=1000,
                              dimensions=fs.n_basis,
                              radius=np.sqrt(fs.n_basis))
        ff_y = nengo.Ensemble(n_neurons=1000,
                              dimensions=fs.n_basis,
                              radius=np.sqrt(fs.n_basis))

        nengo.Connection(net.assoc_mem_x.output[:fs.n_basis], ff_x, synapse=.01)
        nengo.Connection(net.assoc_mem_y.output[:fs.n_basis], ff_y, synapse=.01)

        # -------------------- Product for decoding -----------------------

        net.product_x = nengo.Network('Product X')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=net.product_x,
                               input_magnitude=2.0)
        net.product_y = nengo.Network('Product Y')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=net.product_y,
                               input_magnitude=2.0)

        # get the largest basis function value for normalization
        max_basis = np.max(fs.basis*fs.scale)
        domain = np.linspace(-1, 1, fs.basis.shape[0])

        for ff, product in zip([ff_x, ff_y], [net.product_x, net.product_y]):
            for ii in range(fs.n_basis):
                # find the value of a basis function at a value of x
                def basis_fn(x, jj=ii):
                    index = int(x[0] * len(domain) / 2.0 + len(domain) / 2.0)
                    index = max(min(index, len(domain) - 1), 0)
                    return fs.basis[index][jj]*fs.scale/max_basis
                # multiply the value of each basis function at x by its weight
                nengo.Connection(ramp, product.B[ii], function=basis_fn)
                nengo.Connection(ff[ii], product.A[ii])

        def relay_func(t, x):
            t = time_func(t)
            if t <= -1:
                return [0, 0]
            return x
        relay = nengo.Node(output=relay_func, size_in=2, size_out=2)

        nengo.Connection(net.product_x.output, relay[0],
                         transform=np.ones((1, fs.n_basis)) * max_basis)
        nengo.Connection(net.product_y.output, relay[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis)

        nengo.Connection(relay[0], net.x.input, synapse=None)
        nengo.Connection(relay[1], net.y.input, synapse=None)

        # -------------------- Output ------------------------------

        net.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(net.x.output, net.output[0], synapse=None)
        nengo.Connection(net.y.output, net.output[1], synapse=None)

        # create a node to give a plot of the represented function
        ff_plot = fs.make_plot_node(domain=domain, lines=2,
                                    min_y=-75, max_y=75)
        nengo.Connection(ff_x, ff_plot[:fs.n_basis], synapse=0.1)
        nengo.Connection(ff_y, ff_plot[fs.n_basis:], synapse=0.1)

    return net
