import numpy as np


atts = np.load('./data/attypes.npy', allow_pickle=True)

uatoms = np.unique(np.concatenate(atts))

reps = np.load('./data/aslatms.npy', allow_pickle=True)
reps = [np.array(rep) for rep in reps]

at_envs = {}
# print(reps)
for at in uatoms:

    at_is = [np.where(
        np.array(att) == at)[0] for att in atts]
    # print(at_is)
    at_envs[at] = np.vstack([rep[at_is[i], :] for i, rep in enumerate(
        reps) if len(at_is[i]) > 0])

at_vars = {at: np.var(xx, axis=0) for at, xx in at_envs.items()}
at_n0 = {at: np.where(xx > 0)[0] for at, xx in at_vars.items()}

for at in uatoms:

    at_envs[at] = np.array(at_envs[at])
    at_envs[at] = at_envs[at][:, at_n0[at]]

np.save('./data/red_at_envs', at_envs)
np.save('./data/at_vars', at_vars)
