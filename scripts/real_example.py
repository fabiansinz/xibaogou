import glob
import sys

sys.path.insert(0, '../')
import h5py

from pipeline.utils import CellLabeler
from xibaogou.preprocessing import histeq, unsharp_masking, medianfilter, center, local_standardize

import numpy as np
from xibaogou import RDBP
import pickle


def max_iterator(P, n, voxel):
    P = np.array(P) # copy
    i, j, k = [i // 2 for i in voxel]
    counter = 0

    while counter < n:
        cell = np.where(P == P.max())
        x, y, z = cell[0][0], cell[1][0], cell[2][0]
        P[x - i:x + i, y - j:y + j, z - k:z + k] = -1
        yield (x, y, z)
        counter += 1

preprocess1 = lambda x: histeq(unsharp_masking(medianfilter(center(x))), 500)
preprocess2 = lambda x: local_standardize(medianfilter(x))

linear_channels, quadratic_channels, exponentials = 2, 2, 2
voxel = (17, 17, 15)

rdbp = RDBP(voxel, linear_channels=linear_channels, quadratic_channels=quadratic_channels, exponentials=exponentials)

stacks = []
cells = []
for stack_file in sys.argv[1:]:
    with h5py.File(stack_file,'r') as fid:
        stacks.append(preprocess1(np.asarray(fid['stack'])))
        # stacks.append(np.asarray(fid['stack']))
        cells.append(np.asarray(fid['cells'], dtype=int)-1) # -1 because of matlab to python index conversion

# CellLabeler(stacks[0], cells=cells[0])
rdbp.fit(stacks, cells, maxiter=20)

with open('mymodel.pkl', 'wb') as fid:
    fid.write(rdbp, fid)

P = rdbp.P(stacks[0], full=True)
new_cells = np.asarray(list(max_iterator(P, 100, voxel)), dtype=int)
