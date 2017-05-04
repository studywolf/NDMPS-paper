import numpy as np
import importlib
# import itertools
import time

import abr_control
import nengo

import poppy_config

import ndmps_fs_rhythmic_spa_frontend
importlib.reload(ndmps_fs_rhythmic_spa_frontend)
from ndmps_fs_rhythmic_spa_frontend import generate

input_signals = ['WALKING', 'RUNNING', 'GALLOP']
# input_signals = [" + ".join(x) for x in
#                  itertools.combinations(input_signals, 2)]
# print(input_signals)

poppy = poppy_config.robot_config()
interface = abr_control.interfaces.vrep(poppy)
interface.connect()
interface.disconnect()
interface.connect()
interface.send_target_angles(np.zeros(6))

# for input_signal in input_signals:
#
input_signal = 'WALKING'
print(input_signal)
model = nengo.Network(seed=10)
with model:
    ndmps_r = generate(input_signal=input_signal)
    # output = nengo.Node(
    #     lambda t: np.array([np.sin(t), np.sin(t), np.sin(t),
    #                         np.sin(t), np.sin(t), np.sin(t)]) * 100)
    def vrep_func(t, x):
        # convert to radians
        x[0] *= -1  # ankles abduction are mirrored in VREP
        x[4] *= -1  # hips abduction are mirrored in VREP
        x *= (np.pi / 180.0)
        # the send angles func is expecting (in radians)
        # [l ankle, r ankle, l knee, r knee, l hip, r hip]
        interface.send_target_angles(x)
        # time.sleep(.001)
        return x * 180.0 / np.pi

    node_vrep = nengo.Node(vrep_func, size_in=poppy.num_joints*2, size_out=6)
    nengo.Connection(ndmps_r.output, node_vrep)
        # nengo.Connection(output, node_vrep)
        # probe = nengo.Probe(ndmps_r.output, synapse=.01)
        # probe_x_neurons = nengo.Probe(
        #     ndmps_d.x.y.neurons, synapse=None)
        # probe_y_neurons = nengo.Probe(
        #     ndmps_d.y.y.neurons, synapse=None)

    # sim = nengo.Simulator(model)
    # sim.run(4)
    #
    # # format input string to be appropriate file name
    # input_signal = input_signal.lower().replace(' ', '')
    # np.savez_compressed('results/data/time_steps', sim.trange())
    # np.savez_compressed('results/data/data_%s' % input_signal, sim.data[probe])
    # # np.savez_compressed('results/data/data_%s_x_neurons' % input_signal,
    # #                     sim.data[probe_x_neurons])
    # # np.savez_compressed('results/data/data_%s_y_neurons' % input_signal,
    # #                     sim.data[probe_y_neurons])

# interface.disconnect()
