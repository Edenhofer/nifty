# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_allclose

import nifty6 as ift

from ..common import list2fixture


def _get_rtol(tp):
    if (tp == np.float64) or (tp == np.complex128):
        return 1e-10
    else:
        return 1e-5


space = list2fixture([ift.RGSpace(128)])
sigma = list2fixture([0., .5, 5.])
tp = list2fixture([np.float64, np.complex128])
pmp = pytest.mark.parametrize


def test_property(space, sigma):
    op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
    if op.domain[0] != space:
        raise TypeError


def test_adjoint_times(space, sigma):
    op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
    rand1 = ift.Field.from_random('normal', domain=space)
    rand2 = ift.Field.from_random('normal', domain=space)
    tt1 = rand1.vdot(op.times(rand2))
    tt2 = rand2.vdot(op.adjoint_times(rand1))
    assert_allclose(tt1, tt2)


def test_times(space, sigma):
    op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
    fld = np.zeros(space.shape, dtype=np.float64)
    fld[0] = 1.
    rand1 = ift.Field.from_raw(space, fld)
    tt1 = op.times(rand1)
    assert_allclose(1, tt1.sum())


@pmp('sz', [128, 256])
@pmp('d', [1, 0.4])
def test_smooth_regular1(sz, d, sigma, tp):
    tol = _get_rtol(tp)
    sp = ift.RGSpace(sz, distances=d)
    smo = ift.HarmonicSmoothingOperator(sp, sigma=sigma)
    inp = ift.Field.from_random(
        domain=sp, random_type='normal', std=1, mean=4, dtype=tp)
    out = smo(inp)
    assert_allclose(inp.sum(), out.sum(), rtol=tol, atol=tol)


@pmp('sz1', [10, 15])
@pmp('sz2', [7, 10])
@pmp('d1', [1, 0.4])
@pmp('d2', [2, 0.3])
def test_smooth_regular2(sz1, sz2, d1, d2, sigma, tp):
    tol = _get_rtol(tp)
    sp = ift.RGSpace([sz1, sz2], distances=[d1, d2])
    smo = ift.HarmonicSmoothingOperator(sp, sigma=sigma)
    inp = ift.Field.from_random(
        domain=sp, random_type='normal', std=1, mean=4, dtype=tp)
    out = smo(inp)
    assert_allclose(inp.sum(), out.sum(), rtol=tol, atol=tol)
