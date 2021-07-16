import numpy as np
import time
import sys

init_t = time.time()

at = sys.argv[1]

t1 = time.time()

ref_envs = np.load('./data/train_envs.npy',
                   allow_pickle=True).item()[at]

with open('progress_gen__proj_{}.txt'.format(at), 'a') as file:
    file.write('Reps loaded, time: {} \n'.format(time.time() - t1))


def kernel(pp, sig):
    return np.exp(- pp ** 2 / (2 * sig**2))


with open('progress_gen__proj_{}.txt'.format(at), 'a') as file:
    file.write('Starting train products at {} \n'.format(time.time() - init_t))

atom_diffs = np.load('./euclideans/{}_diffs.npy'.format(at), allow_pickle=True)

t1 = time.time()

sigmas = np.linspace(0.1, 10, 10)

sigma_projections = {}

for sigma in sigmas:
    with open('progress_gen__proj_{}.txt'.format(at), 'a') as file:
        file.write('Starting sigma{} at {} \n'.format(sigma,
                                                      time.time() - init_t))

    atom_projections = []

    for i in range(len(atom_diffs)):
        atom_diff = atom_diffs[i]
        if len(atom_diff) > 0:
            atom_projections.append(np.sum(kernel(np.array(atom_diff), sigma),
                                           axis=1))
        else:
            atom_projections.append(np.zeros(len(ref_envs)))

        if i % 100 == 0:
            with open('progress_gen__proj_{}.txt'.format(at), 'a') as file:
                file.write('    Train mol {}, cost: {} \n'.format(
                    i, time.time() - t1))
            t1 = time.time()

    sigma_projections[sigma] = np.array(atom_projections)

np.save('./projections/{}_projections'.format(at), sigma_projections)
