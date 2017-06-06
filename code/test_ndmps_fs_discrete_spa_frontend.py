import numpy as np
import importlib
import itertools

import nengo

from models import ndmps_fs_discrete_spa_frontend
importlib.reload(ndmps_fs_discrete_spa_frontend)
from models.ndmps_fs_discrete_spa_frontend import generate

input_signals = ['ZERO']#, 'ONE', 'TWO', 'THREE', 'FOUR',
                 # 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
# input_signals += [" + ".join(x) for x in
#                  itertools.combinations(input_signals, 2)]

print(input_signals)

for input_signal in input_signals:

    print(input_signal)
    model = nengo.Network(seed=10)
    with model:
        ndmps_d = generate(input_signal=input_signal)
        probe = nengo.Probe(ndmps_d.output, synapse=.01, sample_every=.005)
        # ndmps_d.product_x.sq1.add_neuron_output()
        # ndmps_d.product_y.sq1.add_neuron_output()
        # probe_product_x = nengo.Probe(
        #     ndmps_d.product_x.sq1.neuron_output, synapse=None)
        # probe_product_y = nengo.Probe(
        #     ndmps_d.product_y.sq1.neuron_output, synapse=None)

    sim = nengo.Simulator(model)
    sim.run(4)

    # format input string to be appropriate file name
    input_signal = input_signal.lower().replace(' ', '')
    np.savez_compressed('results/data/discrete_fs/time_steps', sim.trange())
    np.savez_compressed('results/data/discrete_fs/data_%s' % input_signal, sim.data[probe])
    np.savez_compressed('results/data/discrete_fs/data_%s_x_neurons' % input_signal,
                        sim.data[probe_product_x])
    np.savez_compressed('results/data/discrete_fs/data_%s_y_neurons' % input_signal,
                        sim.data[probe_product_y])
