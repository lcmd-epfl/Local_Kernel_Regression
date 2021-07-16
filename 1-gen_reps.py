import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ase
import ase.io as aio
# from metric_lear<n import MLKR
# import metric_learn
import os
import qml

mol_files = os.listdir('./xyzs')

mols = [aio.read('./xyzs/' + ll) for ll in mol_files]

ncharges = [mol.get_atomic_numbers() for mol in mols]
attypes = [mol.get_chemical_symbols() for mol in mols]
uatoms = np.unique(np.concatenate(attypes))

mbtypes = qml.representations.get_slatm_mbtypes(ncharges)

aslatms = [qml.representations.generate_slatm(
    coordinates=mol.positions,
    nuclear_charges=mol.get_atomic_numbers(),
    mbtypes=mbtypes, local=True) for mol in mols]

np.save('./data/aslatms', aslatms)
np.save('./data/ncharges', ncharges)
np.save('./data/attypes', attypes)
np.savetxt('uatoms.txt', np.array(uatoms))
