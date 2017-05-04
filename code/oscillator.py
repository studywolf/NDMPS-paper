import numpy as np

import nengo


def generate(model, n_neurons=500, speed=.05):
    with model:
        # ------------------ Oscillator -------------------
        osc = nengo.Ensemble(n_neurons=n_neurons, dimensions=2,
                             radius=.9)
        # recurrent connections
        nengo.Connection(osc, osc,
                          transform=np.eye(2) + \
                                    np.array([[1, -1],
                                              [1, 1]]) * speed)
        return osc

if __name__ == '__main__':

    model = nengo.Network('Oscillator')
    with model:
        osc = generate(model=model)
        probe_osc = nengo.Probe(osc, synapse=.1)

    sim = nengo.Simulator(model)
    sim.run(1.5) # run for 1 seconds

    import matplotlib.pyplot as plt
    import seaborn

    plt.subplot(2, 1, 1)
    plt.plot(sim.data[probe_osc][:,0], sim.data[probe_osc][:,1])
    plt.ylabel('state 0')
    plt.xlabel('state 1')

    plt.subplot(2, 1, 2)
    t = sim.trange()
    plt.plot(t, sim.data[probe_osc][:,0])
    plt.plot(t, sim.data[probe_osc][:,1])
    plt.legend(['state 0', 'state 1'])
    plt.ylabel('state')
    plt.xlabel('time (s)')

    plt.tight_layout()
    plt.savefig('oscillator.pdf', format='pdf')
    plt.show()
