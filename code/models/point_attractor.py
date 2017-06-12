""" Implementing ddy = alpha * (beta * (y* - y) - dy)

NOTE: when connecting to the input, use synapse=None so that double
filtering of the input signal doesn't happen. """

import numpy as np
from scipy.linalg import expm

import nengo

def generate(net=None, n_neurons=200, alpha=1000.0, beta=1000.0/4.0,
             dt=0.001, analog=False):
    tau = 0.1  # synaptic time constant

    # the A matrix for our point attractor
    A = np.array([[0.0, 1.0],
                  [-alpha*beta, -alpha]])

    # the B matrix for our point attractor
    B = np.array([[0.0, 0.0], [alpha*beta, 1.0]])

    # discretize
    Ad = expm(A*dt)
    Bd = np.dot(np.linalg.inv(A), np.dot((Ad - np.eye(2)), B))
    # account for discrete lowpass filter
    a = np.exp(-dt/tau)
    if analog:
        A = tau * A + np.eye(2)
        B = tau * B
    else:
        A = 1.0 / (1.0 - a) * (Ad - a * np.eye(2))
        B = 1.0 / (1.0 - a) * Bd

    if net is None:
        net = nengo.Network(label='Point Attractor')
    config = nengo.Config(nengo.Connection, nengo.Ensemble)
    config[nengo.Connection].synapse = nengo.Lowpass(tau)
    # config[nengo.Ensemble].neuron_type = nengo.Direct()

    with config, net:
        net.ydy = nengo.Ensemble(n_neurons=n_neurons, dimensions=2,
            # set it up so neurons are tuned to one dimensions only
            encoders=nengo.dists.Choice([[1, 0], [-1, 0], [0, 1], [0, -1]]))
        # set up Ax part of point attractor
        nengo.Connection(net.ydy, net.ydy, transform=A)

        # hook up input
        net.input = nengo.Node(size_in=2)
        # set up Bu part of point attractor
        nengo.Connection(net.input, net.ydy, transform=B)

        # hook up output
        net.output = nengo.Node(size_in=1)
        # add in forcing function
        nengo.Connection(net.ydy[0], net.output, synapse=None)

    return net


if __name__ == '__main__':

    time = 5  # number of seconds to run simulation
    model = nengo.Network()
    probe_results = []
    for option in [True, False]:
        with model:
            def goal_func(t):
                return [float(int(t)) / time * 2 - 1, 0]
            goal = nengo.Node(output=goal_func)
            pa = generate(n_neurons=1000, analog=option)
            nengo.Connection(goal, pa.input, synapse=None)

            probe_ans = nengo.Probe(goal)
            probe = nengo.Probe(pa.output, synapse=.01)

        sim = nengo.Simulator(model, dt=.001)
        sim.run(time)
        probe_results.append(np.copy(sim.data[probe]))

    import matplotlib.pyplot as plt
    plt.plot(sim.trange(), probe_results[0])
    plt.plot(sim.trange(), probe_results[1], 'g')
    plt.plot(sim.trange(), sim.data[probe_ans][:, 0], 'r--')
    plt.legend(['continuous', 'discrete', 'desired'])
    plt.show()
