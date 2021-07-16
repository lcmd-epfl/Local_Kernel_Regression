import numpy as np
import pandas as pd
import os
import ase
import ase.io as aio

reps = np.load('./data/aslatms.npy', allow_pickle=True)
atts = np.load('./data/attypes.npy', allow_pickle=True)

uatoms = np.unique(np.concatenate(atts))

at_envs = {}

at_vars = np.load('./data/at_vars.npy', allow_pickle=True).item()

at_n0 = {at: np.where(xx > 0)[0] for at, xx in at_vars.items()}

red_reps = []
red_reps_dict = []

for i, rep in enumerate(reps):

    red_rep = [rep[j][at_n0[at]] for j, at in enumerate(atts[i])]

    red_reps.append(red_rep)

    red_rep_dict = {at: np.array(rep)[np.where(np.array(atts[i]) == at)[0]][:, at_n0[at]] for at in uatoms}

    red_reps_dict.append(red_rep_dict)

np.save('./data/red_reps', red_reps)
np.save('./data/red_reps_dict', red_reps_dict)
