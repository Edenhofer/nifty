# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose

import nifty6 as ift

pmp = pytest.mark.parametrize


def _lin2grad(lin):
    return lin.jac(ift.full(lin.domain, 1.)).val


def jt(lin, check):
    assert_allclose(_lin2grad(lin), check)


def test_special_gradients():
    dom = ift.UnstructuredDomain((1,))
    f = ift.full(dom, 2.4)
    var = ift.Linearization.make_var(f)
    s = f.val

    jt(var.clip(0, 10), np.ones_like(s))
    jt(var.clip(-1, 0), np.zeros_like(s))

    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f).sinc()), np.zeros(s.shape))
    assert_(np.isnan(_lin2grad(ift.Linearization.make_var(0*f).absolute())))
    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f + 10).absolute()),
        np.ones(s.shape))
    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f - 10).absolute()),
        -np.ones(s.shape))


@pmp('f', [
    'log', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'sinc', 'sinh', 'cosh', 'tanh',
    'absolute', 'one_over', 'sigmoid', 'log10', 'log1p', "expm1"
])
def test_actual_gradients(f):
    dom = ift.UnstructuredDomain((1,))
    fld = ift.full(dom, 2.4)
    eps = 1e-8
    var0 = ift.Linearization.make_var(fld)
    var1 = ift.Linearization.make_var(fld + eps)
    f0 = getattr(var0, f)().val.val
    f1 = getattr(var1, f)().val.val
    df0 = (f1 - f0)/eps
    df1 = _lin2grad(getattr(var0, f)())
    assert_allclose(df0, df1, rtol=100*eps)
