import importlib
# import matplotlib.pyplot as plt
# import numpy as np

import nengo

import ndmps_discrete
importlib.reload(ndmps_discrete)
from ndmps_discrete import generate

target = .75
model = nengo.Network()
with model:
    ndmps_r = generate(data_file='trajectories/3.npz')
    probe = nengo.Probe(ndmps_r.output, synapse=.1)

sim = nengo.Simulator(model)
sim.run(2)

t = sim.trange()
# plt.figure(figsize=(7, 3.5))
# plt.plot(t, sim.data[probe])
# plt.plot(t, np.ones(t.shape[0]) * target, 'r--', lw=2)
# plt.legend(['position', 'target position'])
# plt.ylabel('state')
# plt.xlabel('time (s)')
#
# plt.tight_layout()
# plt.savefig('point_attractor.pdf', format='pdf')
# plt.show()
