import importlib
import nengo

import forcing_functions
import goal_network
importlib.reload(goal_network)
from goal_network import generate

forces, _, goals = forcing_functions.load_folder('trajectories', rhythmic=False)

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

# sim = nengo.Simulator(model)
# sim.run(2)
#
# t = sim.trange()
# plt.figure(figsize=(7, 3.5))
# plt.plot(t, sim.data[probe_pos])
# plt.plot(t, np.ones(t.shape[0]) * target, 'r--', lw=2)
# plt.legend(['position', 'target position'])
# plt.ylabel('state')
# plt.xlabel('time (s)')
#
# plt.tight_layout()
# plt.savefig('point_attractor.pdf', format='pdf')
# plt.show()
