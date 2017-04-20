import numpy as np

import nengo
import nengo.utils.function_space

import forcing_functions
import oscillator
import point_attractor

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace


def generate(data_folder, net=None):

    # generate the Function Space
    trajectories, _, _ = forcing_functions.load_folder(
        data_folder, rhythmic=True)
    print(np.array(trajectories).shape)
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
        time_func = lambda t: min(max((t * 1) % 6 - 4, -1), 1)

        net.input = nengo.Node(size_in=2, size_out=2, label='input')

        number = nengo.Node(output=[0], label='number')

        # ------------------- Point Attractors --------------------
        x = point_attractor.generate(net.input[0], n_neurons=1000)
        y = point_attractor.generate(net.input[1], n_neurons=1000)

        # -------------------- Oscillators ----------------------
        kick = nengo.Node(nengo.utils.functions.piecewise({0: 1, .05: 0}),
                          label='kick')
        osc = oscillator.generate(net, n_neurons=3000, speed=.005)
        osc.label = 'oscillator'
        nengo.Connection(kick, osc[0])

        # ------------------- Forcing Functions --------------------

        def dmp_weights_func(t, x):
            x = int(min(max(x, 0), len(trajectories)))
            # load weights for generating this number's x and y forces
            return np.hstack([weights_x[x], weights_y[x]])

        # create input switch for generating weights for different numbers
        dmp_weights_gen = nengo.Node(output=dmp_weights_func,
                                     size_in=1,
                                     size_out=fs.n_basis * 2,
                                     label='dmp weights gen')
        nengo.Connection(number, dmp_weights_gen)

        # n_basis_functions dimensions to represent the weights, + 1 to
        # represent the x position to decode from
        ff_x = nengo.Ensemble(n_neurons=1000,
                              dimensions=fs.n_basis,
                              radius=np.sqrt(fs.n_basis),
                              label='ff x')
        ff_y = nengo.Ensemble(n_neurons=1000,
                              dimensions=fs.n_basis,
                              radius=np.sqrt(fs.n_basis),
                              label='ff y')
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
        domain = np.linspace(-np.pi, np.pi, fs.basis.shape[0])
        domain_cossin = np.array([np.cos(domain), np.sin(domain)]).T
        # print(domain)
        for ff, product in zip([ff_x, ff_y], [product_x, product_y]):
            for ii in range(fs.n_basis):
                # find the value of a basis function at a value of (x, y)
                target_function = nengo.utils.connection.target_function(
                    domain_cossin, fs.basis[:, ii]*fs.scale/max_basis)
                nengo.Connection(osc, product.B[ii], **target_function)
                # multiply the value of each basis function at x by its weight
                nengo.Connection(ff[ii], product.A[ii])

        def relay_func(t, x):
            t = time_func(t)
            if t < -1:
                return [0, 0]
            return x
        relay = nengo.Node(output=relay_func, size_in=2, size_out=2,
                           label='relay gate')

        nengo.Connection(product_x.output, relay[0],
                         transform=np.ones((1, fs.n_basis)) * max_basis)
        nengo.Connection(product_y.output, relay[1],
                         transform=np.ones((1, fs.n_basis)) * max_basis)

        nengo.Connection(relay[0], x.input, synapse=None)
        nengo.Connection(relay[1], y.input, synapse=None)

        # -------------------- Output ------------------------------

        net.output = nengo.Node(size_in=2, size_out=2, label='output')
        nengo.Connection(x.output, net.output[0], synapse=None)
        nengo.Connection(y.output, net.output[1], synapse=None)

        # create a node to give a plot of the represented function
        ff_plot = fs.make_plot_node(domain=domain, lines=2,
                                    min_y=-50, max_y=50)
        nengo.Connection(ff_x, ff_plot[:fs.n_basis], synapse=0.1)
        nengo.Connection(ff_y, ff_plot[fs.n_basis:], synapse=0.1)

    return net
