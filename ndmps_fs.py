import numpy as np

import nengo
import nengo.utils.function_space

from point_attractor import gen_point_attractor
from forcing_functions import gen_forcing_functions

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

n_domain_samples = 200
domain = np.linspace(-1, 1, n_domain_samples)

# we know the kinds of forces we want to represent,
# namely, those that will let us draw numbers
goals = []
force_space = []
for ii in range(10):
    y_des = np.load('trajectories/%i.npz' % ii)['arr_0']
    goals.append([y_des[0], y_des[-1]])
    forces, _ = gen_forcing_functions(y_des, num_samples=n_domain_samples)
    force_space.append(forces)
# make an array out of all the possible functions we want to represent
force_space = np.vstack(force_space)

# use this array as our space to perform svd over
fs = nengo.FunctionSpace(space=force_space, n_basis=6)

model = nengo.Network()
with model:

    time_func = lambda t: (t * 2) % 5 - 4
    number = nengo.Node(output=[0])
    # ------------------- Point Attractors --------------------

    def goal_func(t, x):
        t = time_func(t)
        if t < -1:
            return goals[int(x)][0]
        return goals[int(x)][1]
    goal = nengo.Node(output=goal_func, size_in=1)
    nengo.Connection(number, goal)

    x = gen_point_attractor(model, goal[0], n_neurons=500)
    y = gen_point_attractor(model, goal[1], n_neurons=500)

    # -------------------- Ramp ------------------------------
    def ramp_func(t):
        t = time_func(t)
        if t < -1:
            return 0
        return t
    ramp_node = nengo.Node(output=ramp_func)
    ramp = nengo.Ensemble(n_neurons=1000, dimensions=1)
    nengo.Connection(ramp_node, ramp)

    # ------------------- Forcing Functions --------------------

    def dmp_weights_func(t, x):
        x = min(max(x, 0), 9)
        forces = force_space[int(x)*2:int(x)*2+2]
        # load up the forces to be output by the forcing function
        # calculate the corresponding weights over the basis functions
        weights_x = np.dot(fs.basis.T, forces[0])
        weights_y = np.dot(fs.basis.T, forces[1])
        return np.hstack([weights_x, weights_y])

    # create input switch for generating weights for different numbers
    dmp_weights_gen = nengo.Node(output=dmp_weights_func,
                                 size_in=1,
                                 size_out=fs.n_basis * 2)
    nengo.Connection(number, dmp_weights_gen)

    # n_basis_functions dimensions to represent the weights, + 1 to
    # represent the x position to decode from
    ff_x = nengo.Ensemble(n_neurons=500,
                          dimensions=fs.n_basis,
                          radius=np.sqrt(fs.n_basis))
    ff_y = nengo.Ensemble(n_neurons=500,
                          dimensions=fs.n_basis,
                          radius=np.sqrt(fs.n_basis))
    # hook up input
    nengo.Connection(dmp_weights_gen[:fs.n_basis], ff_x)
    nengo.Connection(dmp_weights_gen[fs.n_basis:], ff_y)

    # -------------------- Product for decoding -----------------------

    product_x = nengo.Network('Product X')
    nengo.networks.Product(n_neurons=500,
                           dimensions=fs.n_basis,
                           net=product_x)
    product_y = nengo.Network('Product Y')
    nengo.networks.Product(n_neurons=500,
                           dimensions=fs.n_basis,
                           net=product_y)

    # get the largest basis function value for normalization
    max_basis = np.max(fs.basis*fs.scale)
    for ff, product in zip([ff_x, ff_y], [product_x, product_y]):
        for ii in range(fs.n_basis):
            # function to generate find the value of the
            # basis function at a specified value of x
            def basis_fn(x, jj=ii):
                index = int(x[0]*100+100)
                if index > 199:
                    index = 199
                if index < 0:
                    index = 0
                return fs.basis[index][jj]*fs.scale/max_basis
            # multiply the value of each basis function at x by its weight
            nengo.Connection(ramp, product.B[ii], function=basis_fn)
            nengo.Connection(ff[ii], product.A[ii])

    def relay_func(t, x):
        t = time_func(t)
        if t < -1:
            return [0, 0]
        return x
    relay = nengo.Node(output=relay_func, size_in=2, size_out=2)

    nengo.Connection(product_x.output, relay[0],
                     transform=np.ones((1, fs.n_basis)) * max_basis)
    nengo.Connection(product_y.output, relay[1],
                     transform=np.ones((1, fs.n_basis)) * max_basis)

    nengo.Connection(relay[0], x[1], synapse=None)
    nengo.Connection(relay[1], y[1], synapse=None)

    # -------------------- Output ------------------------------

    output = nengo.Ensemble(n_neurons=1, dimensions=2,
                            neuron_type=nengo.Direct())
    nengo.Connection(x[0], output[0], synapse=.01)
    nengo.Connection(y[0], output[1], synapse=.01)

    # create a node to give a plot of the represented function
    ff_plot = fs.make_plot_node(domain=domain, lines=2, min_y=-30, max_y=30)
    nengo.Connection(ff_x, ff_plot[:fs.n_basis], synapse=0.1)
    nengo.Connection(ff_y, ff_plot[fs.n_basis:], synapse=0.1)


if __name__ == '__main__':
    import nengo_gui
    nengo_gui.Viz(__file__).start()
