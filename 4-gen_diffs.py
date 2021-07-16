import numpy as np
import time
import sklearn as sk
from sklearn import metrics
import gc
import sys
import pickle

init_t = time.time()

at = sys.argv[1]

ref_envs = np.load('./data/train_envs.npy',
                   allow_pickle=True).item()[at]

reps_dict = np.load('./data/red_reps_dict.npy', allow_pickle=True)

attypes = np.load('./data/attypes.npy', allow_pickle=True)

gc.collect()

t1 = time.time()


with open('progress_gen__diffs_{}.txt'.format(at), 'a') as file:
    file.write('Reps loaded, time: {} \n'.format(time.time() - t1))


atom_projections = []

with open('progress_gen__diffs_{}.txt'.format(at), 'a') as file:
    file.write('Starting train products at {} \n'.format(time.time() - init_t))

atom_diffs = []
t1 = time.time()

for i in range(len(reps_dict[:])):
    repd = reps_dict[i]

    rep_at_envs = repd[at]

    if len(rep_at_envs) > 0:

        atom_diff = sk.metrics.pairwise_distances(
            rep_at_envs, ref_envs, n_jobs=-1).T

    else:
        atom_diff = []

    atom_diffs.append(atom_diff)

    if i % 100 == 0:
        with open('progress_gen__diffs_{}.txt'.format(at), 'a') as file:
            file.write('    Train mol {}, cost: {} \n'.format(
                i, time.time() - t1))
        t1 = time.time()


with open('./euclideans/{}_diffs.npy'.format(at), "wb") as fp:
    pickle.dump(atom_diffs, fp)
