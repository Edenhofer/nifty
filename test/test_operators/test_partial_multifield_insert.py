# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose

import nifty6 as ift

from ..common import list2fixture

pmp = pytest.mark.parametrize
dtype = list2fixture([np.float64, np.float32, np.complex64, np.complex128])


def test_part_mf_insert():
    dom = ift.RGSpace(3)
    op1 = ift.ScalingOperator(dom, 1.32).ducktape('a').ducktape_left('a1')
    op2 = ift.ScalingOperator(dom, 1).exp().ducktape('b').ducktape_left('b1')
    op3 = ift.ScalingOperator(dom, 1).sin().ducktape('c').ducktape_left('c1')
    op4 = ift.ScalingOperator(dom, 1).ducktape('c0').ducktape_left('c')**2
    op5 = ift.ScalingOperator(dom, 1).tan().ducktape('d0').ducktape_left('d')
    a = op1 + op2 + op3
    b = op4 + op5
    op = a.partial_insert(b)
    fld = ift.from_random('normal', op.domain)
    ift.extra.check_jacobian_consistency(op, fld)
    assert_(op.domain is ift.MultiDomain.union(
        [op1.domain, op2.domain, op4.domain, op5.domain]))
    assert_(op.target is ift.MultiDomain.union(
        [op1.target, op2.target, op3.target, op5.target]))
    x, y = fld.val, op(fld).val
    assert_allclose(y['a1'], x['a']*1.32)
    assert_allclose(y['b1'], np.exp(x['b']))
    assert_allclose(y['c1'], np.sin(x['c0']**2))
    assert_allclose(y['d'], np.tan(x['d0']))
