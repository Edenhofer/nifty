# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from numpy.testing import assert_allclose, assert_equal

import nifty6 as ift


def test_simplification():
    from nifty6.operators.operator import _ConstantOperator
    f1 = ift.Field.full(ift.RGSpace(10), 2.)
    op = ift.FFTOperator(f1.domain)
    _, op2 = op.simplify_for_constant_input(f1)
    assert_equal(isinstance(op2, _ConstantOperator), True)
    assert_allclose(op(f1).val, op2(f1).val)

    dom = {"a": ift.RGSpace(10)}
    f1 = ift.full(dom, 2.)
    op = ift.FFTOperator(f1.domain["a"]).ducktape("a")
    _, op2 = op.simplify_for_constant_input(f1)
    assert_equal(isinstance(op2, _ConstantOperator), True)
    assert_allclose(op(f1).val, op2(f1).val)

    dom = {"a": ift.RGSpace(10), "b": ift.RGSpace(5)}
    f1 = ift.full(dom, 2.)
    pdom = {"a": ift.RGSpace(10)}
    f2 = ift.full(pdom, 2.)
    o1 = ift.FFTOperator(f1.domain["a"])
    o2 = ift.FFTOperator(f1.domain["b"])
    op = (o1.ducktape("a").ducktape_left("a") +
          o2.ducktape("b").ducktape_left("b"))
    _, op2 = op.simplify_for_constant_input(f2)
    assert_equal(isinstance(op2._op1, _ConstantOperator), True)
    assert_allclose(op(f1)["a"].val, op2(f1)["a"].val)
    assert_allclose(op(f1)["b"].val, op2(f1)["b"].val)
    lin = ift.Linearization.make_var(ift.MultiField.full(op2.domain, 2.), True)
    assert_allclose(op(lin).val["a"].val,
                    op2(lin).val["a"].val)
    assert_allclose(op(lin).val["b"].val,
                    op2(lin).val["b"].val)
