import glob
import sys

sys.path.insert(0, '../')
import h5py

from pipeline.utils import CellLabeler
from xibaogou.preprocessing import histeq, unsharp_masking, medianfilter, center, local_standardize

import numpy as np
from xibaogou import RDBP

preprocess1 = lambda x: histeq(unsharp_masking(medianfilter(center(x))), 500)
preprocess2 = lambda x: local_standardize(medianfilter(x))

linear_channels, quadratic_channels, exponentials = 2, 2, 2
voxel = (17, 17, 15)

rdbp = RDBP(voxel, linear_channels=linear_channels, quadratic_channels=quadratic_channels, exponentials=1)

stacks = []
cells = []
for stack_file in glob.glob(sys.argv[1]):
    with h5py.File(stack_file,'r') as fid:
        stacks.append(preprocess1(np.asarray(fid['stack'])))
        # stacks.append(np.asarray(fid['stack']))
        cells.append(np.asarray(fid['cells'], dtype=int)-1) # -1 because of matlab to python index conversion

# CellLabeler(stacks[0], cells=cells[0])
rdbp.fit(stacks, cells)

