import importlib
import nengo

from models import forcing_functions
from models import goal_network
importlib.reload(goal_network)
from models.goal_network import generate

forces, _, goals = forcing_functions.load_folder(
    'models/handwriting_trajectories', rhythmic=False)

model = nengo.Network()
with model:
    number = nengo.Node(output=[0])
    num_relay = nengo.Node(output=lambda t, x: x / (len(goals) / 2.0) - 1,
                           size_in=1, size_out=1)
    timer_node = nengo.Node(lambda t: min(max((t * 1) % 4 - 2.5, -1), 1))
    gn = generate(goals)
    nengo.Connection(number, num_relay)
    nengo.Connection(num_relay, gn.input)
    nengo.Connection(timer_node, gn.inhibit_node)
    probe = nengo.Probe(gn.output)
