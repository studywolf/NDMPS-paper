""" Implementing ddy = alpha * (beta * (y* - y) - dy)

NOTE: when connecting to the input, use synapse=None so that double
filtering of the input signal doesn't happen. """

import numpy as np

import nengo

def generate(net=None, n_neurons=200, alpha=10.0, beta=10.0/4.0,
             dt=0.001, analog=False):
    tau = 0.05  # synaptic time constant
    synapse = nengo.Lowpass(tau)

    # the A matrix for our point attractor
    a = np.exp(-dt/tau)
    A = np.array([[0.0, 1.0],
                  [-alpha*beta, -alpha]])

    # the B matrix for our point attractor
    B = np.array([[0.0, 0.0], [alpha*beta, 1.0]])

    if analog is False:
        from nengolib.synapses import ss2sim
        C = np.eye(2)
        D = np.zeros((2, 2))
        linsys = ss2sim((A, B, C, D), synapse=synapse, dt=dt)
        A = linsys.A
        B = linsys.B

        # and then compensate for the discrete lowpass filter of synapse
        A = 1.0 / (1.0 - a) * (A - a * np.eye(2))
        B = 1.0 / (1.0 - a) * B

    else:
        A = tau * A + np.eye(2)
        B = tau * B


    if net is None:
        net = nengo.Network(label='Point Attractor')
    config = nengo.Config(nengo.Connection, nengo.Ensemble)
    config[nengo.Connection].synapse = synapse
    # config[nengo.Ensemble].neuron_type = nengo.Direct()

    with config, net:
        net.ydy = nengo.Ensemble(n_neurons=n_neurons, dimensions=2,
            # set it up so neurons are tuned to one dimensions only
            encoders=nengo.dists.Choice([[1, 0], [-1, 0], [0, 1], [0, -1]]))
        # set up Ax part of point attractor
        nengo.Connection(net.ydy, net.ydy, transform=A)

        # hook up input
        net.input = nengo.Node(size_in=2, size_out=2)
        # set up Bu part of point attractor
        nengo.Connection(net.input, net.ydy, transform=B)

        # hook up output
        net.output = nengo.Node(size_in=1, size_out=1)
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
                return [int(t) / time * 2 - 1, 0]
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
