import numpy as np

import nengo
import nengo.utils.function_space

import forcing_functions
import point_attractor

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace


def generate(data_folder, net=None):

    # generate the Function Space
    forces, _, goals = forcing_functions.load_folder(
        data_folder, rhythmic=False)
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

    if net is None:
        net = nengo.Network()
    # net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    with net:

        time_func = lambda t: min(max((t * 1) % 4.5 - 2.5, -1), 1)
        net.number = nengo.Node(size_in=1, size_out=1)
        # ------------------- Point Attractors --------------------

        def goal_func(t, x):
            t = time_func(t)
            if t <= -1:
                return [0, 0]
            return goals[int(x)]
        goal = nengo.Node(output=goal_func, size_in=1, size_out=2)
        nengo.Connection(net.number, goal)

        x = point_attractor.generate(goal[0], n_neurons=1000)
        y = point_attractor.generate(goal[1], n_neurons=1000)

        # -------------------- Ramp ------------------------------
        ramp_node = nengo.Node(output=time_func)
        ramp = nengo.Ensemble(n_neurons=1000, dimensions=1, label='ramp')
        nengo.Connection(ramp_node, ramp)

        # ------------------- Forcing Functions --------------------

        def dmp_weights_func(t, x):
            x = int(min(max(x, 0), len(goals)))
            # load weights for generating this number's x and y forces
            return np.hstack([weights_x[x], weights_y[x]])

        # create input switch for generating weights for different numbers
        dmp_weights_gen = nengo.Node(output=dmp_weights_func,
                                     size_in=1,
                                     size_out=fs.n_basis * 2)
        nengo.Connection(net.number, dmp_weights_gen)

        # n_basis_functions dimensions to represent the weights, + 1 to
        # represent the x position to decode from
        ff_x = nengo.Ensemble(n_neurons=1000,
                              dimensions=fs.n_basis,
                              radius=np.sqrt(fs.n_basis))
        ff_y = nengo.Ensemble(n_neurons=1000,
                              dimensions=fs.n_basis,
                              radius=np.sqrt(fs.n_basis))
        # hook up input
        nengo.Connection(dmp_weights_gen[:fs.n_basis], ff_x)
        nengo.Connection(dmp_weights_gen[fs.n_basis:], ff_y)

        # -------------------- Product for decoding -----------------------

        product_x = nengo.Network('Product X')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=product_x)
        product_y = nengo.Network('Product Y')
        nengo.networks.Product(n_neurons=1000,
                               dimensions=fs.n_basis,
                               net=product_y)

        # get the largest basis function value for normalization
        max_basis = np.max(fs.basis*fs.scale)
        domain = np.linspace(-1, 1, fs.basis.shape[0])

        for ff, product in zip([ff_x, ff_y], [product_x, product_y]):
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

        nengo.Connection(product_x.output, relay[0],
                         transform=np.ones((1, fs.n_basis)) * max_basis)
        nengo.Connection(product_y.output, relay[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis)

        nengo.Connection(relay[0], x.input, synapse=None)
        nengo.Connection(relay[1], y.input, synapse=None)

        # -------------------- Output ------------------------------

        net.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(x.output, net.output[0], synapse=None)
        nengo.Connection(y.output, net.output[1], synapse=None)

        # create a node to give a plot of the represented function
        ff_plot = fs.make_plot_node(domain=domain, lines=2,
                                    min_y=-50, max_y=50)
        nengo.Connection(ff_x, ff_plot[:fs.n_basis], synapse=0.1)
        nengo.Connection(ff_y, ff_plot[fs.n_basis:], synapse=0.1)

    return net
