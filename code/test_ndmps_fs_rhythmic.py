import importlib
# import matplotlib.pyplot as plt
# import numpy as np

import nengo

from models import ndmps_fs_rhythmic
importlib.reload(ndmps_fs_rhythmic)
from models.ndmps_fs_rhythmic import generate

model = nengo.Network()
with model:
    number = nengo.Node(output=[0])
    goals = nengo.Node(output=[0, 0])
    ndmps_r = generate(data_folder='models/handwriting_trajectories')
    nengo.Connection(goals, ndmps_r.input)
    probe = nengo.Probe(ndmps_r.output, synapse=.1)
