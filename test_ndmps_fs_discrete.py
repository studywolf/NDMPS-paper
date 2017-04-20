import importlib
# import matplotlib.pyplot as plt
# import numpy as np

import nengo

import ndmps_fs_discrete
importlib.reload(ndmps_fs_discrete)
from ndmps_fs_discrete import generate

model = nengo.Network()
with model:
    number = nengo.Node(output=[0])
    ndmps_d = generate(data_folder='trajectories')
    nengo.Connection(number, ndmps_d.number)
    probe = nengo.Probe(ndmps_d.output, synapse=.1)

# sim = nengo.Simulator(model)
# sim.run(2)
#
# t = sim.trange()
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
