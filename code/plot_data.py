import matplotlib.pyplot as plt
import numpy as np
import sys

import nengo

folder = sys.argv[1]
num = sys.argv[2]
print(folder)
print(num)

start_time = 300
end_time = 800
n_neurons = 30
time = np.load('results/data/%s/time_steps.npz' % folder)['arr_0'][start_time:end_time]
data = np.load('results/data/%s/data_%s.npz' % (folder, num))['arr_0'][start_time:end_time]
data_x_neurons = np.load(
    'results/data/%s/data_%s_x_neurons.npz' % (folder, num))['arr_0'][start_time:end_time, :n_neurons]
data_y_neurons = np.load(
    'results/data/%s/data_%s_y_neurons.npz' % (folder, num))['arr_0'][start_time:end_time, :n_neurons]

import nengo.utils.matplotlib
nengo.utils.matplotlib.rasterplot(
    time, spikes=data_x_neurons, ax=plt.subplot(2, 1, 1))
plt.xlabel('ff x')
plt.ylabel('time (ms)')
nengo.utils.matplotlib.rasterplot(
    time, spikes=data_y_neurons, ax=plt.subplot(2, 1, 2))
plt.xlabel('ff y')
plt.ylabel('time (ms)')
plt.tight_layout()
plt.savefig('results/%s/spike_rasters/%s_spikes' % (folder, num))  # save png
plt.savefig('results/%s/spike_rasters/%s_spikes.pdf' % (folder, num))  # save pdf

plt.figure()
plt.plot(data[:, 0], data[:, 1])
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.axes().set_aspect('equal')
plt.tight_layout()
plt.savefig('results/%s/pngs/%s' % (folder, num))  # save png
plt.savefig('results/%s/%s.pdf' % (folder, num))  # save pdf


plt.show()
