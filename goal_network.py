import numpy as np

import nengo
import nengo.utils.function_space


def generate(goals, n_neurons=2000, net=None):

    # generate the Function Space
    range_goals = np.array(range(len(goals)))

    if net is None:
        net = nengo.Network()
    with net:

        net.input = nengo.Node(size_in=1, size_out=1)

        def inhibit_func(t, x):
            if abs(x + 1) < 1e-5:
                return [0, 1]
            return [1, 0]
        net.inhibit_node = nengo.Node(output=inhibit_func,
                                      size_in=1, size_out=2,
                                      label='inhibitor')

        goal_on = nengo.Ensemble(n_neurons=n_neurons,
                                 dimensions=1,
                                 label='goal on',
                                 neuron_type=nengo.LIF())
        goal_off = nengo.Ensemble(n_neurons=n_neurons,
                                  dimensions=1,
                                  label='goal off',
                                  neuron_type=nengo.LIF())

        nengo.Connection(net.input, goal_on)
        nengo.Connection(net.input, goal_off)

        net.output = nengo.Node(size_in=2, size_out=2)

        # inhibit on or off population output
        nengo.Connection(net.inhibit_node[0], goal_on.neurons,
                         transform=[[-3]]*goal_on.n_neurons)
        nengo.Connection(net.inhibit_node[1], goal_off.neurons,
                         transform=[[-3]]*goal_on.n_neurons)

        def goal_onoff_func(x, on=True):
            num = range_goals[min(
                max(np.abs(range_goals - ((x+1)*len(goals)/2.0)).argmin(),
                    0),
                len(goals))]
            if on is True:
                return goals[num][0]
            return goals[num][1]
        # hook up on and off populations to output
        nengo.Connection(goal_on, net.output,
                         function=lambda x: goal_onoff_func(x, on=True))
        nengo.Connection(goal_off, net.output,
                         function=lambda x: goal_onoff_func(x, on=False))

    return net
