import numpy as np
import nengo

# from point_attractor import gen_point_attractor_net
import point_attractor
reload(point_attractor)
# from oscillator import gen_oscillator
import oscillator
reload(oscillator)
# from forcing_functions import gen_forcing_functions
import forcing_functions
reload(forcing_functions)

model = nengo.Network()
with model:
    # --------------------- Inputs --------------------------
    goal_x = nengo.Node(output=[0])
    goal_y = nengo.Node(output=[0])

    # ------------------- Point Attractors --------------------
    x = point_attractor.gen_point_attractor_net(model, goal_x, n_neurons=500)
    y = point_attractor.gen_point_attractor_net(model, goal_y, n_neurons=500)

    # -------------------- Oscillators ----------------------
    kick = nengo.Node(nengo.utils.functions.piecewise({0: 1, .05: 0}))
    osc = oscillator.gen_oscillator(model, n_neurons=2000, speed=.01)
    nengo.Connection(kick, osc[0])

    # generate our forcing function
    y_des = np.load('trajectories/heart.npz')['arr_0']
    _, force_data = forcing_functions.gen_forcing_functions(y_des, rhythmic=True)

    # connect oscillator to point attractor
    nengo.Connection(osc, x.input, **force_data[0])
    nengo.Connection(osc, y.input, **force_data[1])

    # output for easy viewing
    output = nengo.Ensemble(n_neurons=1, dimensions=2,
                            neuron_type=nengo.Direct())
    nengo.Connection(x.output, output[0], synapse=.01)
    nengo.Connection(y.output, output[1], synapse=.01)

if __name__ == '__main__':
    import nengo_gui
    nengo_gui.Viz(__file__).start()
