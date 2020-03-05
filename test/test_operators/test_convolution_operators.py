# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from numpy.testing import assert_allclose

import nifty6 as ift
import numpy as np

from ..common import list2fixture

space = list2fixture([
    ift.RGSpace(4),
    ift.HPSpace(4),
    ift.GLSpace(4)
])


def test_const_func(space):
    sig = ift.Field.from_random('normal', domain=space)
    fco_op = ift.FuncConvolutionOperator(space, lambda x: np.ones(x.shape))
    vals = fco_op(sig).val
    vals = np.round(vals, decimals=5)
    assert len(np.unique(vals)) == 1


def gauss(x, sigma):
    normalization = np.sqrt(2. * np.pi) * sigma
    return np.exp(-0.5 * x * x / sigma**2) / normalization


def test_gaussian_smoothing():
    N = 128
    sigma = N / 10**4
    dom = ift.RGSpace(N)
    sig = ift.exp(ift.Field.from_random('normal', dom))
    fco_op = ift.FuncConvolutionOperator(dom, lambda x: gauss(x, sigma))
    sm_op = ift.HarmonicSmoothingOperator(dom, sigma)
    assert_allclose(fco_op(sig).val,
                    sm_op(sig).val,
                    rtol=1e-05)
    assert_allclose(fco_op.adjoint_times(sig).val,
                    sm_op.adjoint_times(sig).val,
                    rtol=1e-05)
