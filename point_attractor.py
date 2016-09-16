import numpy as np

import nengo


def gen_point_attractor_net(model, goal, n_neurons=200, alpha=10, beta=10/4.):
    # create a network with point attractor dynamics
    # separating out the velocity and position into different dimensions
    # to more efficiently optimize their radii
    with model:
        pa_net = nengo.Network()
        pa_net.config[nengo.Connection].synapse = nengo.Lowpass(1)
        with pa_net:
            pa_net.input = nengo.Node(size_in=1, size_out=1)
            y = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                               radius=5)
            dy = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                                radius=30)
            pa_net.output = nengo.Node(size_in=1, size_out=1)
            # set up recurrent connection for system state
            nengo.Connection(y, y)
            nengo.Connection(dy, dy)
            # set up connections to implement point attractor
            nengo.Connection(dy, y)
            nengo.Connection(y, dy,
                             transform=-alpha*beta)
            nengo.Connection(dy, dy,
                             transform=-alpha)
            nengo.Connection(goal, dy,
                             transform=alpha*beta)
            # hook up input and output
            nengo.Connection(pa_net.input, dy, synapse=None)
            nengo.Connection(y, pa_net.output, synapse=None)

    return pa_net


if __name__ == '__main__':

    target = .75
    model = nengo.Network('Point attractor')
    with model:
        goal = nengo.Node(output=[target])
        pa = gen_point_attractor_net(model=model, goal=goal)
        probe_pos = nengo.Probe(pa.output, synapse=.1)

    sim = nengo.Simulator(model)
    sim.run(2) # run for 5 seconds

    import matplotlib.pyplot as plt
    import seaborn

    t = sim.trange()
    plt.figure(figsize=(7,3.5))
    plt.plot(t, sim.data[probe_pa])
    plt.plot(t, np.ones(t.shape[0]) * target, 'r--', lw=2)
    plt.legend(['position', 'target position'])
    plt.ylabel('state')
    plt.xlabel('time (s)')

    plt.tight_layout()
    plt.savefig('Figure1-point_attractor.pdf', format='pdf')
    plt.show()
