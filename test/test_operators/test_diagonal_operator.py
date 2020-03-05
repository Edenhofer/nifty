# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from numpy.testing import assert_allclose, assert_equal

import nifty6 as ift

from ..common import list2fixture

space = list2fixture([
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])


def test_property(space):
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    if D.domain[0] != space:
        raise TypeError


def test_times_adjoint(space):
    rand1 = ift.Field.from_random('normal', domain=space)
    rand2 = ift.Field.from_random('normal', domain=space)
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    tt1 = rand1.vdot(D.times(rand2))
    tt2 = rand2.vdot(D.times(rand1))
    assert_allclose(tt1, tt2)


def test_times_inverse(space):
    rand1 = ift.Field.from_random('normal', domain=space)
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    tt1 = D.times(D.inverse_times(rand1))
    assert_allclose(rand1.val, tt1.val)


def test_times(space):
    rand1 = ift.Field.from_random('normal', domain=space)
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    tt = D.times(rand1)
    assert_equal(tt.domain[0], space)


def test_adjoint_times(space):
    rand1 = ift.Field.from_random('normal', domain=space)
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    tt = D.adjoint_times(rand1)
    assert_equal(tt.domain[0], space)


def test_inverse_times(space):
    rand1 = ift.Field.from_random('normal', domain=space)
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    tt = D.inverse_times(rand1)
    assert_equal(tt.domain[0], space)


def test_adjoint_inverse_times(space):
    rand1 = ift.Field.from_random('normal', domain=space)
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    tt = D.adjoint_inverse_times(rand1)
    assert_equal(tt.domain[0], space)


def test_diagonal(space):
    diag = ift.Field.from_random('normal', domain=space)
    D = ift.DiagonalOperator(diag)
    diag_op = D(ift.Field.full(space, 1.))
    assert_allclose(diag.val, diag_op.val)
