============================================
NDMPs paper
============================================

This is a repo for neural Dynamic Movement Primitives.

To run the files in this folder, you'll need the function-space-4 branches
of Nengo and Nengo GUI. Once installed, open the test_* files to play around
with them.

To load in new trajectories, save a (movement_length, 2) vector with

```
import numpy as np
np.savez_compressed(filename, trajectory_vector)
```

in the trajectories folder. The function space scripts will auto-load this
file, and the vector space scripts can load it by name.
