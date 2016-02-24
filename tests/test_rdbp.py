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


def test_separable_convolution():
    """Testing separable convolution"""
    channels = 3
    flt_row, flt_col, flt_depth = (7, 5, 3)
    in_channels = 1

    b = RDBP((flt_row, flt_col, flt_depth), quadratic_channels=channels, linear_channels=channels)
    X = np.random.randn(50, 40, 30)
    # X_ = th.shared(np.require(X, dtype=floatX), borrow=True, name='stack')
    X_ = T.tensor3(dtype=floatX)

    c_, (xy_, z_) = b._build_separable_convolution(channels, X_, X.shape)
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


def test_exponent():
    linear_channels, quadratic_channels, common_channels = 4, 3, 2
    flt_row, flt_col, flt_depth = (11, 7, 3)

    Uxy = np.random.randn(quadratic_channels, flt_row, flt_col)
    Uz = np.random.randn(quadratic_channels, flt_depth)
    Wxy = np.random.randn(linear_channels, flt_row, flt_col)
    Wz = np.random.randn(linear_channels, flt_depth)
    beta = np.random.randn(common_channels, quadratic_channels)
    gamma = np.random.randn(common_channels, linear_channels)
    b = np.random.randn(common_channels)

    rdbp = RDBP((flt_row, flt_col, flt_depth), quadratic_channels=quadratic_channels, linear_channels=linear_channels,
                common_channels=2)
    X = np.random.randn(50, 40, 30)
    X_ = T.tensor3(dtype=floatX)
    quadratic_filter_, (Uxy_, Uz_) = rdbp._build_separable_convolution(quadratic_channels, X_, X.shape)
    linear_filter_, (Wxy_, Wz_) = rdbp._build_separable_convolution(linear_channels, X_, X.shape)
    beta_ = T.dmatrix()
    gamma_ = T.dmatrix()

    squadr_filter_ = T.tensordot(beta_, quadratic_filter_ ** 2, (1, 0))  # .dimshuffle(3, 0, 1, 2)
    slin_filter_ = T.tensordot(gamma_, linear_filter_, (1, 0))  # .dimshuffle(3, 0, 1, 2)

    qf = th.function([X_, Uxy_, Uz_], quadratic_filter_)
    lf = th.function([X_, Wxy_, Wz_], linear_filter_)

    qf2 = th.function([X_, Uxy_, Uz_, beta_], squadr_filter_)
    lf2 = th.function([X_, Wxy_, Wz_, gamma_], slin_filter_)

    exponent_, (Uxy_, Uz_, Wxy_, Wz_, beta_, gamma_, b_) = rdbp._build_exponent(X_, X.shape)
    ef = th.function([X_, Uxy_, Uz_, Wxy_, Wz_, beta_, gamma_, b_], exponent_)

    Q = qf(X, Uxy, Uz)
    L = lf(X, Wxy, Wz)

    Q2 = np.tensordot(beta, Q ** 2, (1, 0))
    L2 = np.tensordot(gamma, L, (1, 0))

    Qs = qf2(X, Uxy, Uz, beta)
    Ls = lf2(X, Wxy, Wz, gamma)

    assert_true(np.abs(L2 - Ls).max() < 1e-10, 'linear part does not match up')
    assert_true(np.abs(Q2 - Qs).max() < 1e-10, 'quadratic part does not match up')

    expo = Q2 + L2 + b[:, None, None, None]
    assert_true(np.abs(ef(X, Uxy, Uz, Wxy, Wz, beta, gamma, b) - expo).max() < 1e-10, 'exponent does not match up')


def test_probability():
    linear_channels, quadratic_channels, common_channels = 4, 3, 2
    flt_row, flt_col, flt_depth = (11, 7, 3)

    Uxy = np.random.randn(quadratic_channels, flt_row, flt_col)
    Uz = np.random.randn(quadratic_channels, flt_depth)
    Wxy = np.random.randn(linear_channels, flt_row, flt_col)
    Wz = np.random.randn(linear_channels, flt_depth)
    beta = np.random.randn(common_channels, quadratic_channels)
    gamma = np.random.randn(common_channels, linear_channels)
    b = np.random.randn(common_channels)

    rdbp = RDBP((flt_row, flt_col, flt_depth), quadratic_channels=quadratic_channels, linear_channels=linear_channels,
                common_channels=2)

    X = np.random.randn(50, 40, 30)
    X_ = T.tensor3(dtype=floatX)
    exponent_, params_ = rdbp._build_exponent(X_, X.shape)
    ef = th.function((X_,) + params_, exponent_)

    e = ef(X, Uxy, Uz, Wxy, Wz, beta, gamma, b)
    p = np.exp(e).sum(axis=0)
    # apply logistic function to log p_ and add a bit of offset for numerical stability
    p = p / (1 + p) * (1 - 2 * 1e-8) + 1e-8



    p_, params_ = rdbp._build_probability_map(X_, X.shape)
    pf = th.function((X_,) + params_, p_)
    p2 = pf(X, Uxy, Uz, Wxy, Wz, beta, gamma, b)
    idx = ~np.isnan(p2) & ~np.isnan(p)
    assert_true(np.abs(p2[idx] - p[idx]).max() < 1e-10, 'probability does not match up')


if __name__ == "__main__":
        test_probability()
