import numpy as np

import nengo
import nengo.utils.function_space

from . import forcing_functions
from . import goal_network
from . import point_attractor

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

def generate(data_folder, net=None, alpha = 1000.0):
    beta = alpha / 4.0

    # generate the Function Space
    forces, _, goals = forcing_functions.load_folder(
        data_folder, rhythmic=False, alpha=alpha, beta=beta)
    # make an array out of all the possible functions we want to represent
    force_space = np.vstack(forces)
    # use this array as our space to perform svd over
    fs = nengo.FunctionSpace(space=force_space, n_basis=10)
    range_goals = np.array(range(len(goals)))

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

        time_func = lambda t: min(max((t * 2) % 4 - 2.5, -1), 1)
        timer_node = nengo.Node(output=time_func)
        net.number = nengo.Node(output=lambda t, x: x / (len(goals) / 2.0) - 1,
                                size_in=1, size_out=1)
        # ------------------- Point Attractors --------------------

        goal_net = goal_network.generate(goals)
        nengo.Connection(net.number, goal_net.input)
        nengo.Connection(timer_node, goal_net.inhibit_node)

        net.x = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)
        nengo.Connection(goal_net.output[0], net.x.input[0], synapse=None)
        net.y = point_attractor.generate(
            n_neurons=1000, alpha=alpha, beta=beta)
        nengo.Connection(goal_net.output[1], net.y.input[0], synapse=None)

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

        def dmp_weights_func(x, x_or_y):
            # find the nearest value
            num = range_goals[min(
                max(np.abs(range_goals - ((x+1)*len(goals)/2.0)).argmin(),
                    0),
                len(goals))]
            # load weights for generating this number's x and y forces
            if x_or_y == 'x':
                return weights_x[num]
            elif x_or_y == 'y':
                return weights_y[num]

        # generate weights for different numbers
        dmp_weights_gen = nengo.Ensemble(n_neurons=2000, dimensions=1,
                                         label='dmp_weights_gen')
        nengo.Connection(net.number, dmp_weights_gen)
        nengo.Connection(dmp_weights_gen, ff_x,
                         function=lambda x: dmp_weights_func(x, x_or_y='x'),
                         synapse=.01)
        nengo.Connection(dmp_weights_gen, ff_y,
                         function=lambda x: dmp_weights_func(x, x_or_y='y'),
                         synapse=.01)

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
                         transform=np.ones((1, fs.n_basis)) * max_basis,
                         synapse=None)
        nengo.Connection(product_y.output, relay[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis,
                         synapse=None)

        nengo.Connection(relay[0], net.x.input[1], synapse=None)
        nengo.Connection(relay[1], net.y.input[1], synapse=None)

        # -------------------- Output ------------------------------

        net.output = nengo.Node(size_in=2)
        nengo.Connection(net.x.output, net.output[0], synapse=0.01)
        nengo.Connection(net.y.output, net.output[1], synapse=0.01)

        # create a node to give a plot of the represented function
        ff_plot = fs.make_plot_node(domain=domain, lines=2,
                                    ylim=[-50, 50])
        nengo.Connection(ff_x, ff_plot[:fs.n_basis], synapse=0.1)
        nengo.Connection(ff_y, ff_plot[fs.n_basis:], synapse=0.1)

    return net
