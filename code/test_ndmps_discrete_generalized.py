import importlib
import numpy as np

import nengo

import ndmps_discrete_generalized
importlib.reload(ndmps_discrete_generalized)
from ndmps_discrete_generalized import generate

# input_signals = range(10)

# for input_signal in input_signals:
model = nengo.Network()
input_signal = 2

# targets = [[1, 0], [1, 1], [0, 1], [-1, 1],
#            [-1, 0], [-1, -1], [0, -1], [1, -1]]
targets = [[.2, -.4]]#, [.5, -1], [1, -2]]

for target in targets:
    with model:

        ndmps, goals = generate(
            data_file='handwriting_trajectories/%s.npz' % input_signal)

        print(goals)
        node_start = nengo.Node(output=[0, 0], label='start goal')
        node_end = nengo.Node(output=target, label='end goal')
        nengo.Connection(node_start, ndmps.goal_transformed[:2])
        nengo.Connection(node_end, ndmps.goal_transformed[2:])
        nengo.Connection(node_start, ndmps.goal_relay[:2])
        nengo.Connection(node_end, ndmps.goal_relay[2:])

        probe = nengo.Probe(ndmps.output, synapse=.01, sample_every=.01)
        probe_x_neurons = nengo.Probe(
            ndmps.x.y.neurons, synapse=None)
        probe_y_neurons = nengo.Probe(
            ndmps.y.y.neurons, synapse=None)

        sim = nengo.Simulator(model)
        sim.run(10)

        name = '%s_%.2f_%.2f' % (input_signal, target[0], target[1])
        # format input string to be appropriate file name
        np.savez_compressed('results/data/discrete_generalized/time_steps', sim.trange())
        np.savez_compressed('results/data/discrete_generalized/data_%s' % name, sim.data[probe])
        np.savez_compressed('results/data/discrete_generalized/data_%s_x_neurons' % name,
                            sim.data[probe_x_neurons])
        np.savez_compressed('results/data/discrete_generalized/data_%s_y_neurons' % name,
                            sim.data[probe_y_neurons])
