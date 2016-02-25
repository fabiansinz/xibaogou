import sys

sys.path.insert(0, '../')
import numpy as np
from xibaogou import RDBP

stack_number = 4
cells_per_stack = 50

linear_channels, quadratic_channels, exponentials = 1, 1, 1
voxel = (3, 3, 3)

rdbp = RDBP(voxel, linear_channels=linear_channels, quadratic_channels=quadratic_channels, exponentials=1)

stack_size = 30, 20, 25

cells = [np.c_[tuple((np.random.randint(i, size=cells_per_stack) for i in stack_size))] for _ in range(stack_number)]
Ys = [rdbp._build_label_stack(stack_size, c, full=True) for c in cells]
stacks = [rdbp._build_label_stack(stack_size, c, full=True) + np.random.randn(*stack_size) * 0.01 for c in cells]

rdbp.fit(stacks, cells)
#----------------------------------
# TODO: Remove this later
from IPython import embed
embed()
# exit()
#----------------------------------
