Generating the LKR model consists on 6 steps:

1- Generating the representations

The script 1-gen_reps.py generates the aSLATM representations of the structures in the xyzs directory

2- Separating the atomic environments of different atomic spicies

The script 2-gen_at_envs.py agregates the the aSLATM representations of atoms of the same type. 
At the same time, it reduces the size of the aSLATM representations by evaluating which dimensions of the representation are constant over the data.

3- Reducing the size of aSLATM representations.

Using the results from 2, aSLATM representations of the training points are reduced by removing the unchanging dimensions.

3.5- The first pool of reference atomic environments must be selected.

The script 3.5-FPS_filter_ref_envs.py selects filters the environments from the training data using a file with the indices of the selected environments.


4- Generate the euclidean distances between the atoms in the training data and the reference atoms
    
Script 4-gen_diffs.py generates the distances for a specific atom type. Execute as:

    python 4-gen_diffs.py atomtype

where atomtype should be the chemical symbols of the atom (e.g. C for carbon)

5- Generate the kernel projection of the training molecules to the reference atoms

Script 5-gen_projections.py generates the projections for a specific atom type. Execute as:

    python 5-gen_projections.py atomtype sigma

where atomtype should be the chemical symbols of the atom (e.g. C for carbon), and sigma is the value of the sigma hyperparameter in the gaussian kernel.


6- Train the model using OMP

