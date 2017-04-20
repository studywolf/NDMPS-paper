""" Implementing ddy = alpha * (beta * (y* - y) - dy) """

# NOTE: need to mark explicitly where expecting synapse=None on
# incoming connections

import nengo


def generate(goal, net=None, n_neurons=200, alpha=10.0, beta=10.0/4.0):
    # create a network with point attractor dynamics
    # separating out the velocity and position into different dimensions
    # to more efficiently optimize their radii
    if net is None:
        net = nengo.Network(label='Point Attractor')
    # NOTE: by setting tau = 1 filtering is easier to account for
    config = nengo.Config(nengo.Connection, nengo.Ensemble)
    config[nengo.Connection].synapse = nengo.Lowpass(1)
    with config, net:
        net.y = nengo.Ensemble(
            n_neurons=n_neurons, dimensions=1, radius=5)
        net.dy = nengo.Ensemble(
            n_neurons=n_neurons, dimensions=1, radius=30)
        # set up recurrent connection for system state
        nengo.Connection(net.y, net.y)
        nengo.Connection(net.dy, net.dy)
        # set up connections to implement point attractor
        nengo.Connection(net.dy, net.y)
        nengo.Connection(net.y, net.dy, transform=-alpha*beta)
        nengo.Connection(net.dy, net.dy, transform=-alpha)
        nengo.Connection(goal, net.dy, transform=alpha*beta)

        # create node for saving the accelerations (everything fed into dy)
        # NOTE: pa_force is just for debugging
        net.pa_force = nengo.Node(size_in=1, size_out=1)
        nengo.Connection(net.dy, net.pa_force)
        nengo.Connection(net.y, net.pa_force, transform=-alpha*beta)
        nengo.Connection(net.dy, net.pa_force, transform=-alpha)
        nengo.Connection(goal, net.pa_force, transform=alpha*beta)

        # hook up input and output
        net.input = nengo.Node(size_in=1, size_out=1)
        net.output = nengo.Node(size_in=1, size_out=1)
        nengo.Connection(net.input, net.dy)
        nengo.Connection(net.y, net.output, synapse=None)

    return net
