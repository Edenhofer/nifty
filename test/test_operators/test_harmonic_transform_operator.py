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


tp = list2fixture([np.float64, np.float32, np.complex64, np.complex128])
lm = list2fixture([128, 256])
pmp = pytest.mark.parametrize


def test_dotsht(lm, tp):
    tol = 10*_get_rtol(tp)
    a = ift.LMSpace(lmax=lm)
    b = ift.GLSpace(nlat=lm + 1)
    fft = ift.HarmonicTransformOperator(domain=a, target=b)
    inp = ift.Field.from_random(
        domain=a, random_type='normal', std=1, mean=0, dtype=tp)
    out = fft.times(inp)
    v1 = np.sqrt(out.vdot(out))
    v2 = np.sqrt(inp.vdot(fft.adjoint_times(out)))
    assert_allclose(v1, v2, rtol=tol, atol=tol)


def test_dotsht2(lm, tp):
    tol = 10*_get_rtol(tp)
    a = ift.LMSpace(lmax=lm)
    b = ift.HPSpace(nside=lm//2)
    fft = ift.HarmonicTransformOperator(domain=a, target=b)
    inp = ift.Field.from_random(
        domain=a, random_type='normal', std=1, mean=0, dtype=tp)
    out = fft.times(inp)
    v1 = np.sqrt(out.vdot(out))
    v2 = np.sqrt(inp.vdot(fft.adjoint_times(out)))
    assert_allclose(v1, v2, rtol=tol, atol=tol)


@pmp('space', [ift.LMSpace(lmax=30, mmax=25)])
def test_normalisation(space, tp):
    tol = 10*_get_rtol(tp)
    cospace = space.get_default_codomain()
    fft = ift.HarmonicTransformOperator(space, cospace)
    inp = ift.Field.from_random(
        domain=space, random_type='normal', std=1, mean=2, dtype=tp)
    out = fft.times(inp)
    zero_idx = tuple([0]*len(space.shape))
    assert_allclose(
        inp.val[zero_idx], out.integrate(), rtol=tol, atol=tol)
