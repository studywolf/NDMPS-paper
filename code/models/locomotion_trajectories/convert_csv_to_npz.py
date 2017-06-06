import glob
import numpy as np

files = glob.glob('*.csv')

for f in files:
    data = np.genfromtxt(f, delimiter=',')
    # the [:-4] to strip off the extension
    np.savez_compressed(f[:-4], data[:, 1][:, None])
