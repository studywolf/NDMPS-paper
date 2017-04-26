import importlib

import nengo

import ndmps_fs_discrete_spa_frontend
importlib.reload(ndmps_fs_discrete_spa_frontend)
from ndmps_fs_discrete_spa_frontend import generate

model = nengo.Network()
with model:
    # number = nengo.Node(output=[0])
    ndmps_d = generate(data_folder='trajectories')
    # nengo.Connection(number, ndmps_d.input)
    probe = nengo.Probe(ndmps_d.output, synapse=.1)
