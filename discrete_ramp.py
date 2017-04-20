import matplotlib.pyplot as plt
import numpy as np
import seaborn

import nengo


model = nengo.Network()
with model:

    node_input = nengo.Node(output=.01)

    integrator = nengo.Ensemble(n_neurons=500, dimensions=1)
    nengo.Connection(integrator, integrator)
    nengo.Connection(node_input, integrator)

    probe_int = nengo.Probe(integrator, synapse=.05)

sim = nengo.Simulator(model)
sim.run(1)

plt.figure(figsize=(6,3))
plt.plot(sim.trange(), sim.data[probe_int])
plt.ylabel('x')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig('discrete_ramp.pdf', format='pdf')
plt.show()
