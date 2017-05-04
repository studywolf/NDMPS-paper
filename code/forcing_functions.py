import glob
import numpy as np
from scipy import interpolate

import nengo

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def generate(y_des, dt=.001, n_samples=1000,
             alpha=10.0, beta=10.0/4.0, rhythmic=False,
             plotting=False, generalized=False):

    # scale our trajectory and find the center point
    center = np.sum(y_des, axis=1) / y_des.shape[1]
    # center trajectory around (0, 0)
    y_des -= center[:, None]
    if rhythmic is True:
        goal = np.zeros(center.shape)
        return_goal = center
    else:
        start = y_des[:, 0]
        goal = y_des[:, -1]

        # if we're generalizing things, rotate the trajectory
        # so that start and end points lie on x-axis
        if generalized == True:
            # move start to (0, 0)
            y_des -= start[:, None]
            start = y_des[:, 0]
            goal = y_des[:, -1]
            # move end to x-axis
            theta = np.arctan2(goal[1], goal[0])
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            y_des = np.dot(R.T, y_des)
            # normalize movement length to 1
            y_des /= y_des[0, -1]
            # get new end point on x-axis
            goal = y_des[:, -1]


    # interpolate our desired trajectory to smooth out the sampling
    path = np.zeros((y_des.shape[0], n_samples))
    x = np.linspace(-1, 1, y_des.shape[1])
    for d in range(y_des.shape[0]):
        path_gen = interpolate.interp1d(x, y_des[d])
        for ii, t in enumerate(np.linspace(-1, 1, n_samples)):
            path[d, ii] = path_gen(t)
    y_des = path

    # calculate velocity of y_des
    dy_des = np.diff(y_des) / dt / n_samples
    # add zero to the beginning of every row
    dy_des = np.hstack((np.zeros((y_des.shape[0], 1)), dy_des))

    # calculate acceleration of y_des
    ddy_des = np.diff(dy_des) / dt / n_samples
    # add zero to the beginning of every row
    ddy_des = np.hstack((np.zeros((y_des.shape[0], 1)), ddy_des))

    forces = []
    pa_forces = []
    forcing_functions = []
    x = np.linspace(-np.pi, np.pi, n_samples)
    for ii in range(y_des.shape[0]):
        # find the force required to move along this trajectory
        # by subtracting out the effects of the point attractor
        pa_forces.append(alpha * (beta * (goal[ii] - y_des[ii]) - dy_des[ii]))

        forces.append(ddy_des[ii] - pa_forces[ii])

        # if generalized is True and rhythmic is False:
        #     forces[ii] /= (goal[ii] - start[ii])

        # create the lookup table set for training oscillator output
        forcing_functions.append(
            nengo.utils.connection.target_function(
                np.array([np.cos(x), np.sin(x)]).T,
                forces[ii]))

    if plotting is True:
        import matplotlib.pyplot as plt

        plt.figure()
        ax = plt.subplot(4, 1, 1)
        plt.plot(y_des[0], y_des[1])
        plt.plot(start[0], start[1], 'bx', mew=3)
        plt.plot(goal[0], goal[1], 'rx', mew=3)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        ax.set_aspect('equal')
        plt.subplot(4, 1, 2)
        plt.plot(y_des[0])
        plt.plot(y_des[1])
        plt.title('position')
        plt.subplot(4, 1, 3)
        plt.plot(dy_des[0])
        plt.plot(dy_des[1])
        plt.title('velocity')
        plt.subplot(4, 1, 4)
        plt.plot(ddy_des[0])
        plt.plot(ddy_des[1])
        plt.title('acceleration')
        plt.tight_layout()

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(pa_forces[0])
        plt.plot(pa_forces[1])
        plt.title('point attractor forces')
        plt.subplot(2, 1, 2)
        plt.plot(forces[0])
        plt.plot(forces[1])
        # plt.plot(-pa_forces[0], alpha=.5)
        # plt.plot(-pa_forces[1], alpha=.5)
        plt.title('forcing function')
        plt.tight_layout()
        plt.show()

    if rhythmic is False:
        goal = [start, goal]
    else:
        goal = return_goal
    return forces, forcing_functions, goal


def load_folder(folder, rhythmic, plotting=False, generalized=False):

    # generate the Function Space
    force_space = []
    force_functions = []
    goals = []
    trajectory_list = sorted(glob.glob(folder + '/*.npz'))
    for ii, data_file in enumerate(trajectory_list):
        print(data_file)
        # print out trajectories list and corresponding numbers
        # read in the trajectories, calculate the kinds of
        # forces needed to generate them, this is our function space
        y_des = np.load(data_file)['arr_0'].T
        forces, force_function, goal = generate(
            y_des, rhythmic=rhythmic, plotting=plotting,
            generalized=generalized)
        force_space.append(forces)
        force_functions.append(force_function)
        goals.append(goal)

    return force_space, force_functions, goals

if __name__ == '__main__':

    load_folder('handwriting_trajectories',
                rhythmic=False, plotting=True,
                generalized=True)
