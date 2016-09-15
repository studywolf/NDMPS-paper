import numpy as np

import nengo


def gen_point_attractor(model, goal, n_neurons=200, alpha=10, beta=10/4.):
    # create an ensemble with point attractor dynamics
    with model:
        yz = nengo.Ensemble(n_neurons=n_neurons, dimensions=2, radius=5,
                neuron_type=nengo.Direct())
        # set up recurrent connection for system state, which
        # specify the rate of change of each dimension represented.
        # first row of the transform is dyz, second row is ddyz
        nengo.Connection(yz, yz,
                         transform=(np.eye(2) +
                                    np.array([[0, 1],
                                              [-alpha*beta, -alpha]])),
                         synapse=1)
        # connect up the input signal
        nengo.Connection(goal, yz[1],
                         transform=[[alpha*beta]],
                         synapse=1)
        return yz

if __name__ == '__main__':

    target = .75
    model = nengo.Network('Point attractor')
    with model:
        goal = nengo.Node(output=[target])
        pa = gen_point_attractor(model=model, goal=goal)
        probe_pa = nengo.Probe(pa, synapse=.1)

    sim = nengo.Simulator(model)
    sim.run(2) # run for 5 seconds

    import matplotlib.pyplot as plt
    import seaborn

    t = sim.trange()
    plt.figure(figsize=(7,3.5))
    plt.plot(t, sim.data[probe_pa])
    plt.plot(t, np.ones(t.shape[0]) * target, 'r--', lw=2)
    plt.legend(['position', 'velocity', 'target position'])
    plt.ylabel('state')
    plt.xlabel('time (s)')

    plt.tight_layout()
    plt.savefig('Figure1-point_attractor.pdf', format='pdf')
    plt.show()
