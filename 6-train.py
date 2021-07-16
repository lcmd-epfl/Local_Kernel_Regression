import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import OrthogonalMatchingPursuit


init_t = time.time()
step = 1

target = np.loadtxt('./target.dat')

attypes = np.load('./data/attypes.npy', allow_pickle=True)

atoms = np.unique(np.concatenate(attypes))

projections = np.concatenate(
    [np.load('/projections/{}_projections.npy'.format(at)) for at in atoms],
    axis=1)

train_indx, test_indx = train_test_split(
    np.arange(len(target)), test_size=0.1, random_state=2)

train_y = target[train_indx]
test_y = target[test_indx]

train_projections = projections[train_indx]
test_projections = projections[test_indx]

# OMP
sizes = np.arange(10, 1000, 100)
# sizes = [5000]
# sizes[0] = 200

test_maes = []
train_maes = []
omp_models = []

with open('progress.txt', 'a') as file:
    file.write('Starting omp trainings {} \n'.format(
        time.time() - init_t))

train_preds_list = []
test_preds_list = []

for size in sizes:
    t1 = time.time()

    reg = OrthogonalMatchingPursuit(
        n_nonzero_coefs=int(size)).fit(train_projections, train_y)

    omp_models.append(reg)

    test_preds = reg.predict(test_projections)
    test_mae = np.mean(np.abs(test_preds - test_y))

    train_preds = reg.predict(train_projections)
    train_mae = np.mean(np.abs(train_preds - train_y))

    test_maes.append(test_mae)
    train_maes.append(train_mae)

    train_preds_list.append(train_preds)
    test_preds_list.append(test_preds)

    with open('progress.txt', 'a') as file:
        file.write(
            'Omp size: {}, cost: {}, train acc: {}, test acc: {}\n'.format(
                size, time.time() - t1, train_mae, test_mae))

np.save('train_preds_list', train_preds_list)
np.save('test_preds_list', test_preds_list)
np.save('omp_models', omp_models)
np.save('test_mae', test_maes)
np.save('train_mae', train_maes)
