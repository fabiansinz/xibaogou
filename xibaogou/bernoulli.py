import os
import numpy as np
import theano as th
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import theano as th
from collections import OrderedDict
from scipy.ndimage import convolve1d

floatX = th.config.floatX
T = th.tensor
import theano.tensor.nnet.conv3d2d
from scipy.special import beta
from sklearn.metrics import roc_auc_score

tensor5 = theano.tensor.TensorType('float64', 5 * [False])


class BP:
    def __init__(self, voxel):

        if np.any(np.array(voxel) % 2 == 0):
            raise ValueError("Voxel size should be odd.")
        self.voxel = voxel
        self.parameters = OrderedDict()

    def _build_label_stack(self, X, cell_locations, full=False):
        """
        Builds that stack in which the locations indicated by cell_locations are set to one.
        Otherwise the values of the stack are zero.

        :param X: original stack
        :param cell_locations: Nx3 integer array will cell locations (0 based indices)
        :param full: indicates whether the result should have full size of the size after valid convolution
        :return: numpy array with stack indicating the cell locations.
        """

        if full:
            Y = np.zeros_like(X)
            i, j, k = cell_locations.T
            Y[i, j, k] = 1
        else:
            y_shape = tuple(i - j + 1 for i, j in zip(X.shape, self.voxel))
            Y = np.zeros(y_shape)

            cell_locations = cell_locations[np.all(cell_locations < Y.shape, axis=1)
                                            & np.all(cell_locations >= 0, axis=1)]
            cell_locs = cell_locations - np.array([v // 2 for v in self.voxel])

            i, j, k = cell_locs.T
            Y[i, j, k] = 1

        return Y

    def _build_crossentropy(self, X, cell_locations):
        Y = self._build_label_stack(X, cell_locations)
        Y_ = th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')

        p_, parameters_ = self._build_probability_map(X)

        loglik_ = Y_ * T.log(p_) + (1 - Y_) * T.log(1 - p_)

        cross_entropy_ = -T.mean(loglik_)
        dcross_entropy_ = T.grad(cross_entropy_, parameters_)

        return th.function(parameters_, cross_entropy_), th.function(parameters_, dcross_entropy_)

    def set_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.parameters:
                self.parameters[k] = v

    def P(self, X, full=False):
        p_, params_ = self._build_probability_map(X)
        p = th.function(params_, p_)
        P = p(*tuple(self.parameters.values()))
        if not full:
            return P
        else:
            fullP = 0 * X
            i, j, k = [(i - j + 1) // 2 for i, j in zip(X.shape, P.shape)]
            fullP[i:-i, j:-j, k:-k] = P
            return fullP

    def auc(self, X, cell_locations, **kwargs):
        return roc_auc_score(self._build_label_stack(X, cell_locations).ravel(), self.P(X).ravel(), **kwargs)

    def cross_entropy(self, X, cell_locations):
        ce, _ = self._build_crossentropy(X, cell_locations)
        return ce(*self.parameters.values()) / np.log(2)

    def fit(self, X, cell_locations, **options):
        ll, dll = self._build_crossentropy(X, cell_locations)
        # p_, params_ = self._build_probability_map(X)
        # P = th.function(params_, p_)
        slices, shapes = [], []
        i = 0
        for elem in self.parameters.values():
            slices.append(slice(i, i + elem.size))
            shapes.append(elem.shape)
            i += elem.size

        def ravel(params):
            return np.hstack([e.ravel() for e in params])

        def unravel(x):
            return tuple(x[sl].reshape(sh) for sl, sh in zip(slices, shapes))

        def obj(x):
            return ll(*unravel(x))

        def dobj(x):
            return ravel(dll(*unravel(x)))

        def callback(x):
            print('Cross entropy:', obj(x))

        x0 = ravel(self.parameters.values())
        # todo find a better way than to box constrain the parameters
        opt_results = minimize(obj, x0, jac=dobj, method='L-BFGS-B', callback=callback,
                               bounds=list(zip(-1000 * np.ones(len(x0)), 1000 * np.ones(len(x0)))),
                               options=options)
        for k, param in zip(self.parameters, unravel(opt_results.x)):
            self.parameters[k] = param


class RDBP(BP):
    def __init__(self, voxel, exponentials=2, linear_channels=2, quadratic_channels=2):
        super(RDBP, self).__init__(voxel)

        self.linear_channels = linear_channels
        self.quadratic_channels = quadratic_channels
        self.common_channels = exponentials
        flt_width, flt_height, flt_depth = self.voxel

        # horizontal components of the filters
        self.parameters['u_xy'] = np.random.rand(quadratic_channels, flt_width, flt_height, 1)
        self.parameters['u_xy'] /= self.parameters['u_xy'].size

        # certial components of the filters
        self.parameters['u_z'] = np.random.rand(quadratic_channels, 1, flt_depth, 1)
        self.parameters['u_z'] /= self.parameters['u_z'].size

        # horizontal components of the filters
        self.parameters['w_xy'] = np.random.rand(linear_channels, flt_width, flt_height, 1)
        self.parameters['w_xy'] /= self.parameters['w_xy'].size

        # vertical components of the filters
        self.parameters['w_z'] = np.random.rand(linear_channels, 1, flt_depth, 1)
        self.parameters['w_z'] /= self.parameters['w_z'].size

        self.parameters['beta'] = np.random.randn(exponentials, quadratic_channels)
        self.parameters['gamma'] = np.random.randn(exponentials, linear_channels)
        self.parameters['b'] = np.random.randn(exponentials)

    def _build_separable_convolution(self, no_of_filters, X_, data_shape):
        """
        Builds a theano function that performas a 3d convolution which is separable in
        xy vs. z on the stack.

        :param no_of_filters: number of convolution filters
        :param X_: 3d tensor representing the stack (row, col, depth)
        :param data_shape: shape of the real data (the tensor has no shape yet)
        :return: theano symbolic expression, (Uxy tensor, Uz tensor)
        """

        # X_ is row, col, depth == in_shape
        Vxy_ = T.tensor3(dtype=floatX)  # filters, row, col
        Vz_ = T.matrix(dtype=floatX)  # filters, depth

        batchsize, in_channels = 1, 1
        in_width, in_height, in_depth = data_shape
        flt_row, flt_col, flt_depth = self.voxel

        # X is row, col, depth, channel
        xy_ = T.nnet.conv2d(
            # expects (batch size, channels, row, col), transform in to (depth, 1, row, col)
            input=X_.dimshuffle(2, 'x', 0, 1),
            # expects nb filters, channels, nb row, nb col
            filters=Vxy_.dimshuffle(0, 'x', 1, 2),
            filter_shape=(no_of_filters, in_channels, flt_row, flt_col),
            image_shape=(in_depth, in_channels, in_width, in_height),
            border_mode='valid'
        ).dimshuffle(1, 2, 3, 0)  # the output is shaped (filters, row, col, depth)

        retval_, _ = theano.map(
            lambda v, f:
            T.nnet.conv2d(
                # v is (row, col, depth) and well make it
                # (row, 1, col, depth) = (batch size, stack size, nb row, nb col)
                input=v.dimshuffle(0, 'x', 1, 2),
                # f is (flt_depth, ) and we'll make it
                # (1, 1, in_channels, flt_depth) =  (nb filters, stack size, nb row, nb col)
                filters=f.dimshuffle('x', 'x', 'x', 0),  # nb filters, stack size, nb row, nb col
                image_shape=(in_width - flt_row + 1, 1, in_height - flt_col + 1, in_depth),
                filter_shape=(1, 1, in_channels, flt_depth),
                border_mode='valid'
            ).squeeze()
            , sequences=(xy_, Vz_))
        return retval_, (Vxy_, Vz_)

    def _build_exponent(self, X_, data_shape):
        """
        Builds the exponent of the nonlinearty (see README or Theis et al. 2013)

        :param X_: 3d tensor representing the stack (row, col, depth)
        :param data_shape: shape of the real data (the tensor has no shape yet)
        :return: symbolic tensor for the exponent, (Uxy, Uz, Wxy, Wz, beta, gamma, b)
        """

        linear_channels, quadratic_channels, common_channels = \
            self.linear_channels, self.quadratic_channels, self.common_channels

        quadratic_filter_, (Uxy_, Uz_) = self._build_separable_convolution(quadratic_channels, X_, data_shape)
        linear_filter_, (Wxy_, Wz_) = self._build_separable_convolution(linear_channels, X_, data_shape)

        b_ = T.dvector()  # bias
        beta_ = T.dmatrix()
        gamma_ = T.dmatrix()

        quadr_filter_ = T.tensordot(beta_, quadratic_filter_ ** 2, (1, 0))
        lin_filter_ = T.tensordot(gamma_, linear_filter_, (1, 0))

        exponent_ = quadr_filter_ + lin_filter_ + b_.dimshuffle(0, 'x', 'x', 'x')
        return exponent_, (Uxy_, Uz_, Wxy_, Wz_, beta_, gamma_, b_)

    def _build_probability_map(self, X_, data_shape):
        """
        Builds a theano symbolic expression that yields the estimated probability of a cell per voxel.


        :param X_: 3d tensor representing the stack (row, col, depth)
        :param data_shape: shape of the real data (the tensor has no shape yet)
        :return: symbolic tensor for P, (Uxy, Uz, Wxy, Wz, beta, gamma, b)
        """

        exponent_, params_ = self._build_exponent(X_, data_shape)
        p_ = T.exp(exponent_).sum(axis=0)
        # apply logistic function to log p_ and add a bit of offset for numerical stability
        p_ = p_ / (1 + p_) * (1 - 2 * 1e-8) + 1e-8
        return p_, params_

    def __str__(self):
        return """
        Range degenerate Bernoulli process

        quadratic components: %i
        linear components: %i
        common components: %i
        """ % (self.quadratic_channels, self.linear_channels, self.common_channels)

    def __repr__(self):
        return self.__str__()
