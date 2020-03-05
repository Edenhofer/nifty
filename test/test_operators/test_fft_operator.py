# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose

import nifty6 as ift

from ..common import list2fixture


def _get_rtol(tp):
    if (tp == np.float64) or (tp == np.complex128):
        return 1e-10
    else:
        return 1e-5


pmp = pytest.mark.parametrize
dtype = list2fixture([np.float64, np.float32, np.complex64, np.complex128])
op = list2fixture([ift.HartleyOperator, ift.FFTOperator])


@pmp('d', [0.1, 1, 3.7])
def test_fft1D(d, dtype, op):
    dim1 = 16
    tol = _get_rtol(dtype)
    a = ift.RGSpace(dim1, distances=d)
    b = ift.RGSpace(dim1, distances=1./(dim1*d), harmonic=True)
    np.random.seed(16)

    fft = op(domain=a, target=b)
    inp = ift.Field.from_random(
        domain=a, random_type='normal', std=7, mean=3, dtype=dtype)
    out = fft.inverse_times(fft.times(inp))
    assert_allclose(inp.val, out.val, rtol=tol, atol=tol)

    a, b = b, a

    fft = ift.FFTOperator(domain=a, target=b)
    inp = ift.Field.from_random(
        domain=a, random_type='normal', std=7, mean=3, dtype=dtype)
    out = fft.inverse_times(fft.times(inp))
    assert_allclose(inp.val, out.val, rtol=tol, atol=tol)


@pmp('dim1', [12, 15])
@pmp('dim2', [9, 12])
@pmp('d1', [0.1, 1, 3.7])
@pmp('d2', [0.4, 1, 2.7])
@pmp('nthreads', [-1, 1, 2, 3, 4])
def test_fft2D(dim1, dim2, d1, d2, dtype, op, nthreads):
    ift.fft.set_nthreads(nthreads)
    assert_(ift.fft.nthreads() == nthreads)
    tol = _get_rtol(dtype)
    a = ift.RGSpace([dim1, dim2], distances=[d1, d2])
    b = ift.RGSpace(
        [dim1, dim2], distances=[1./(dim1*d1), 1./(dim2*d2)], harmonic=True)

    fft = op(domain=a, target=b)
    inp = ift.Field.from_random(
        domain=a, random_type='normal', std=7, mean=3, dtype=dtype)
    out = fft.inverse_times(fft.times(inp))
    assert_allclose(inp.val, out.val, rtol=tol, atol=tol)

    a, b = b, a

    fft = ift.FFTOperator(domain=a, target=b)
    inp = ift.Field.from_random(
        domain=a, random_type='normal', std=7, mean=3, dtype=dtype)
    out = fft.inverse_times(fft.times(inp))
    assert_allclose(inp.val, out.val, rtol=tol, atol=tol)
    ift.fft.set_nthreads(1)


@pmp('index', [0, 1, 2])
def test_composed_fft(index, dtype, op):
    tol = _get_rtol(dtype)
    a = [a1, a2,
         a3] = [ift.RGSpace((32,)),
                ift.RGSpace((4, 4)),
                ift.RGSpace((5, 6))]
    fft = op(domain=a, space=index)

    inp = ift.Field.from_random(
        domain=(a1, a2, a3), random_type='normal', std=7, mean=3, dtype=dtype)
    out = fft.inverse_times(fft.times(inp))
    assert_allclose(inp.val, out.val, rtol=tol, atol=tol)


@pmp('space', [
    ift.RGSpace(128, distances=3.76, harmonic=True),
    ift.RGSpace((15, 27), distances=(.7, .33), harmonic=True),
    ift.RGSpace(73, distances=0.5643)
])
def test_normalisation(space, dtype, op):
    tol = 10*_get_rtol(dtype)
    cospace = space.get_default_codomain()
    fft = op(space, cospace)
    inp = ift.Field.from_random(
        domain=space, random_type='normal', std=1, mean=2, dtype=dtype)
    out = fft.times(inp)
    fft2 = op(cospace, space)
    out2 = fft2.inverse_times(inp)
    zero_idx = tuple([0]*len(space.shape))
    assert_allclose(
        inp.val[zero_idx], out.integrate(), rtol=tol, atol=tol)
    assert_allclose(out.val, out2.val, rtol=tol, atol=tol)
