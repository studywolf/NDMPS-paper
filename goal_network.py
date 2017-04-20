import numpy as np

import nengo
import nengo.utils.function_space


# def generate1(data_folder, net=None):
#
#     # generate the Function Space
#     forces, _, goals = forcing_functions.load_folder(
#         data_folder, rhythmic=False)
#     range_goals = np.array(range(len(goals)))
#
#     if net is None:
#         net = nengo.Network()
#     with net:
#
#         net.input = nengo.Node(output=lambda t, x: x / (len(goals) / 2.0) - 1,
#                                 size_in=1, size_out=1)
#
#         def timer_func(t):
#             t = min(max((t * 1) % 4 - 2.5, -1), 1)
#             if t <= -1:
#                 return -.5
#             return .5
#         timer_node = nengo.Node(output=timer_func, size_out=1, label='timer')
#         num_eval_points = 10000
#         eval_points1 = np.ones(num_eval_points) * .5
#         eval_points1[int(num_eval_points/2):] *= -1
#         eval_points2 = np.random.uniform(-1, 1, num_eval_points)
#         eval_points = np.vstack([eval_points1, eval_points2]).T
#         goal_ens = nengo.Ensemble(n_neurons=1000, dimensions=2,
#                                   radius=np.sqrt(2),
#                                   label='goal gen',
#                                   eval_points=eval_points)
#         nengo.Connection(timer_node, goal_ens[0])
#         nengo.Connection(net.input, goal_ens[1])
#
#         # for debugging --------------------
#         goal_ans = nengo.Ensemble(n_neurons=1, dimensions=2,
#                                   label='goal ans',
#                                   neuron_type=nengo.Direct())
#         nengo.Connection(timer_node, goal_ans[0])
#         nengo.Connection(net.input, goal_ans[1])
#         # ----------------------------------
#
#         def goal_func(x):
#             num = range_goals[min(
#                 max(np.abs(range_goals - ((x[1]+1)*len(goals)/2.0)).argmin(),
#                     0),
#                 len(goals))]
#             if x[0] < 0:
#                 return goals[num][0]
#             return goals[num][1]
#
#         net.output= nengo.Node(size_in=4, size_out=4, label='goal relay')
#         nengo.Connection(goal_ens, net.output[:2],
#                          function=goal_func)
#         nengo.Connection(goal_ans, net.output[2:],
#                          function=goal_func)
#
#     return net

def generate(goals, n_neurons=2000, net=None):

    # generate the Function Space
    range_goals = np.array(range(len(goals)))

    if net is None:
        net = nengo.Network()
    with net:

        net.input = nengo.Node(size_in=1, size_out=1)

        def timer_func(t, x):
            if abs(x + 1) < 1e-5:
                return [0, 1]
            return [1, 0]
        net.timer_node = nengo.Node(output=timer_func,
                                    size_in=1, size_out=2, label='timer')

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

        net.output= nengo.Node(size_in=2, size_out=2)  # set to 4 when debugging

        # inhibit on or off population output
        nengo.Connection(net.timer_node[0], goal_on.neurons,
                         transform=[[-3]]*goal_on.n_neurons)
        nengo.Connection(net.timer_node[1], goal_off.neurons,
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
        nengo.Connection(goal_on, net.output[:2],
                         function=lambda x: goal_onoff_func(x, on=True))
        nengo.Connection(goal_off, net.output[:2],
                         function=lambda x: goal_onoff_func(x, on=False))

        # # for debugging --------------------
        # def timer_func_debug(t):
        #     t = min(max((t * 1) % 4 - 2.5, -1), 1)
        #     if t <= -1:
        #         return -1
        #     return 1
        # timer_node_debug = nengo.Node(output=timer_func_debug,
        #                               size_out=1, label='timer')
        # def goal_func_debug(x):
        #     num = range_goals[min(
        #         max(np.abs(range_goals - ((x[1]+1)*len(goals)/2.0)).argmin(),
        #             0),
        #         len(goals))]
        #     if x[0] < 0:
        #         return goals[num][0]
        #     return goals[num][1]
        # goal_ans = nengo.Ensemble(n_neurons=1, dimensions=2,
        #                           label='goal ans',
        #                           neuron_type=nengo.Direct())
        # nengo.Connection(timer_node_debug, goal_ans[0])
        # nengo.Connection(net.input, goal_ans[1])
        # nengo.Connection(goal_ans, net.output[2:],
        #                  function=goal_func_debug)
        # # ----------------------------------

    return net
