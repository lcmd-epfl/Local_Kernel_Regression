import numpy as np

at_envs = np.load('./data/red_at_envs.npy', allow_pickle=True).item()

# This step requires a prior selection of the reference pool of enviornments.
# indices_atoms = np.load('indices_reference_atoms.npy')

# Alternatively, here we use all the environments
for key in at_envs.keys():
    # at_envs[key] = at_envs[key][indices_atoms[key]]
    at_envs[key] = at_envs[key]

np.save('./data/train_envs', at_envs)
