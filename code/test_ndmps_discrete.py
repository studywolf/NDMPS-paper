import importlib
import numpy as np

import nengo

import ndmps_discrete
importlib.reload(ndmps_discrete)
from ndmps_discrete import generate

input_signals = range(10)

for input_signal in input_signals:
    model = nengo.Network()
    with model:
        ndmps = generate(
            data_file='handwriting_trajectories/%s.npz' % input_signal)
        probe = nengo.Probe(ndmps.output, synapse=.01)
        probe_ramp = nengo.Probe(ndmps.ramp)
        probe_ramp_neurons = nengo.Probe(
            ndmps.ramp.neurons, synapse=None)
        probe_x_neurons = nengo.Probe(
            ndmps.x.y.neurons, synapse=None)
        probe_y_neurons = nengo.Probe(
            ndmps.y.y.neurons, synapse=None)

    sim = nengo.Simulator(model)
    sim.run(4)

    # format input string to be appropriate file name
    np.savez_compressed('results/data/discrete/time_steps', sim.trange())
    np.savez_compressed('results/data/discrete/ramp', sim.data[probe_ramp])
    np.savez_compressed('results/data/discrete/ramp_neurons', sim.data[probe_ramp_neurons])
    np.savez_compressed('results/data/discrete/data_%s' % input_signal, sim.data[probe])
    np.savez_compressed('results/data/discrete/data_%s_x_neurons' % input_signal,
                        sim.data[probe_x_neurons])
    np.savez_compressed('results/data/discrete/data_%s_y_neurons' % input_signal,
                        sim.data[probe_y_neurons])
