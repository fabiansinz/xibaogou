import sys

sys.path.append('../')
from nose.tools import assert_true
import numpy as np
from xibaogou import RDBP
import theano as th

floatX = th.config.floatX
T = th.tensor
import theano.tensor.nnet.conv3d2d
from scipy import signal
import itertools


def test_convolution():
    """Testing separable convolution"""
    channels = 3
    flt_row, flt_col, flt_depth = (7, 5, 3)
    in_channels = 1

    b = RDBP((flt_row, flt_col, flt_depth), quadratic_channels=channels, linear_channels=channels)
    X = np.random.randn(50, 40, 30)
    # X_ = th.shared(np.require(X, dtype=floatX), borrow=True, name='stack')

    c_, (xy_, z_), X_ = b._build_separable_convolution(channels, X.shape)
    xy = np.random.randn(channels, flt_row, flt_col)
    z = np.random.randn(channels, flt_depth)
    f = th.function([X_, xy_, z_], c_)

    for k in range(channels):
        inter1 = [signal.convolve2d(e, xy[k, ...].squeeze(), mode='valid') for e in X.squeeze().transpose([2, 0, 1])]
        inter1 = np.stack(inter1, axis=2)
        inter2 = 0 * inter1[..., flt_depth - 1:]

        for i, j in itertools.product(range(44), range(36)):
            inter2[i, j, :] = signal.convolve(inter1[i, j, :], z[k].squeeze(), mode='valid')
        assert_true(np.abs(inter2 - f(X, xy, z)[k, ...]).max() < 1e-10)


if __name__ == "__main__":
    test_convolution()
