import matplotlib.pyplot as plt
import numpy as np

import nengo

from point_attractor import generate

target = .75
model = nengo.Network()
with model:
    goal = nengo.Node(output=[target])
    pa = generate(goal=goal)
    probe_pos = nengo.Probe(pa.output, synapse=.1)
    probe_pa_force = nengo.Probe(pa.pa_force, synapse=None)

sim = nengo.Simulator(model)
sim.run(2)

t = sim.trange()
plt.figure(figsize=(7, 3.5))
plt.plot(t, sim.data[probe_pos])
plt.plot(t, np.ones(t.shape[0]) * target, 'r--', lw=2)
plt.legend(['position', 'target position'])
plt.ylabel('state')
plt.xlabel('time (s)')

plt.tight_layout()
plt.savefig('point_attractor.pdf', format='pdf')
plt.show()
