# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import nifty6 as ift

dom = ift.makeDomain({"d1": ift.RGSpace(10)})


def test_vdot():
    f1 = ift.from_random("normal", domain=dom, dtype=np.complex128)
    f2 = ift.from_random("normal", domain=dom, dtype=np.complex128)
    assert_allclose(f1.vdot(f2), np.conj(f2.vdot(f1)))


def test_func():
    f1 = ift.from_random("normal", domain=dom, dtype=np.complex128)
    assert_allclose(
        ift.log(ift.exp((f1)))["d1"].val, f1["d1"].val)


def test_multifield_field_consistency():
    f1 = ift.full(dom, 27)
    f2 = ift.makeField(dom['d1'], f1['d1'].val)
    assert_equal(f1.sum(), f2.sum())
    assert_equal(f1.size, f2.size)


def test_dataconv():
    f1 = ift.full(dom, 27)
    f2 = ift.makeField(dom, f1.val)
    for key, val in f1.items():
        assert_equal(val.val, f2[key].val)
    if "d1" not in f2:
        raise KeyError()
    assert_equal({"d1": f1}, f2.to_dict())
    f3 = ift.full(dom, 27+1.j)
    f4 = ift.full(dom, 1.j)
    assert_equal(f2, f3.real)
    assert_equal(f4, f3.imag)


def test_blockdiagonal():
    op = ift.BlockDiagonalOperator(
        dom, {"d1": ift.ScalingOperator(dom["d1"], 20.)})
    op2 = op(op)
    ift.extra.consistency_check(op2)
    assert_equal(type(op2), ift.BlockDiagonalOperator)
    f1 = op2(ift.full(dom, 1))
    for val in f1.values():
        assert_equal((val == 400).all(), True)
    op2 = op + op
    assert_equal(type(op2), ift.BlockDiagonalOperator)
    f1 = op2(ift.full(dom, 1))
    for val in f1.values():
        assert_equal((val == 40).all(), True)
