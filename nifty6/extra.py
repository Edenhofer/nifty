# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from numpy.testing import assert_

from .domain_tuple import DomainTuple
from .field import Field
from .linearization import Linearization
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.linear_operator import LinearOperator
from .sugar import from_random

__all__ = ["consistency_check", "check_jacobian_consistency",
           "assert_allclose"]


def assert_allclose(f1, f2, atol, rtol):
    if isinstance(f1, Field):
        return np.testing.assert_allclose(f1.val, f2.val, atol=atol, rtol=rtol)
    for key, val in f1.items():
        assert_allclose(val, f2[key], atol=atol, rtol=rtol)


def _adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol,
                            only_r_linear):
    needed_cap = op.TIMES | op.ADJOINT_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    f1 = from_random("normal", op.domain, dtype=domain_dtype)
    f2 = from_random("normal", op.target, dtype=target_dtype)
    res1 = f1.vdot(op.adjoint_times(f2))
    res2 = op.times(f1).vdot(f2)
    if only_r_linear:
        res1, res2 = res1.real, res2.real
    np.testing.assert_allclose(res1, res2, atol=atol, rtol=rtol)


def _inverse_implementation(op, domain_dtype, target_dtype, atol, rtol):
    needed_cap = op.TIMES | op.INVERSE_TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    foo = from_random("normal", op.target, dtype=target_dtype)
    res = op(op.inverse_times(foo))
    assert_allclose(res, foo, atol=atol, rtol=rtol)

    foo = from_random("normal", op.domain, dtype=domain_dtype)
    res = op.inverse_times(op(foo))
    assert_allclose(res, foo, atol=atol, rtol=rtol)


def _full_implementation(op, domain_dtype, target_dtype, atol, rtol,
                         only_r_linear):
    _adjoint_implementation(op, domain_dtype, target_dtype, atol, rtol,
                            only_r_linear)
    _inverse_implementation(op, domain_dtype, target_dtype, atol, rtol)


def _check_linearity(op, domain_dtype, atol, rtol):
    needed_cap = op.TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    fld1 = from_random("normal", op.domain, dtype=domain_dtype)
    fld2 = from_random("normal", op.domain, dtype=domain_dtype)
    alpha = np.random.random()  # FIXME: this can break badly with MPI!
    val1 = op(alpha*fld1+fld2)
    val2 = alpha*op(fld1)+op(fld2)
    assert_allclose(val1, val2, atol=atol, rtol=rtol)


def _actual_domain_check(op, domain_dtype=None, inp=None):
    needed_cap = op.TIMES
    if (op.capability & needed_cap) != needed_cap:
        return
    if domain_dtype is not None:
        inp = from_random("normal", op.domain, dtype=domain_dtype)
    elif inp is None:
        raise ValueError('Need to specify either dtype or inp')
    assert_(inp.domain is op.domain)
    assert_(op(inp).domain is op.target)


def _actual_domain_check_nonlinear(op, loc):
    assert isinstance(loc, (Field, MultiField))
    assert_(loc.domain is op.domain)
    lin = Linearization.make_var(loc, False)
    reslin = op(lin)
    assert_(lin.domain is op.domain)
    assert_(lin.target is op.domain)
    assert_(lin.val.domain is lin.domain)

    assert_(reslin.domain is op.domain)
    assert_(reslin.target is op.target)
    assert_(reslin.val.domain is reslin.target)

    assert_(reslin.target is op.target)
    assert_(reslin.jac.domain is reslin.domain)
    assert_(reslin.jac.target is reslin.target)
    _actual_domain_check(reslin.jac, inp=loc)
    _actual_domain_check(reslin.jac.adjoint, inp=reslin.jac(loc))


def _domain_check(op):
    for dd in [op.domain, op.target]:
        if not isinstance(dd, (DomainTuple, MultiDomain)):
            raise TypeError(
                'The domain and the target of an operator need to',
                'be instances of either DomainTuple or MultiDomain.')


def consistency_check(op, domain_dtype=np.float64, target_dtype=np.float64,
                      atol=0, rtol=1e-7, only_r_linear=False):
    """
    Checks an operator for algebraic consistency of its capabilities.

    Checks whether times(), adjoint_times(), inverse_times() and
    adjoint_inverse_times() (if in capability list) is implemented
    consistently. Additionally, it checks whether the operator is linear.

    Parameters
    ----------
    op : LinearOperator
        Operator which shall be checked.
    domain_dtype : dtype
        The data type of the random vectors in the operator's domain. Default
        is `np.float64`.
    target_dtype : dtype
        The data type of the random vectors in the operator's target. Default
        is `np.float64`.
    atol : float
        Absolute tolerance for the check. If rtol is specified,
        then satisfying any tolerance will let the check pass.
        Default: 0.
    rtol : float
        Relative tolerance for the check. If atol is specified,
        then satisfying any tolerance will let the check pass.
        Default: 0.
    only_r_linear: bool
        set to True if the operator is only R-linear, not C-linear.
        This will relax the adjointness test accordingly.
    """
    if not isinstance(op, LinearOperator):
        raise TypeError('This test tests only linear operators.')
    _domain_check(op)
    _actual_domain_check(op, domain_dtype)
    _actual_domain_check(op.adjoint, target_dtype)
    _actual_domain_check(op.inverse, target_dtype)
    _actual_domain_check(op.adjoint.inverse, domain_dtype)
    _check_linearity(op, domain_dtype, atol, rtol)
    _check_linearity(op.adjoint, target_dtype, atol, rtol)
    _check_linearity(op.inverse, target_dtype, atol, rtol)
    _check_linearity(op.adjoint.inverse, domain_dtype, atol, rtol)
    _full_implementation(op, domain_dtype, target_dtype, atol, rtol,
                         only_r_linear)
    _full_implementation(op.adjoint, target_dtype, domain_dtype, atol, rtol,
                         only_r_linear)
    _full_implementation(op.inverse, target_dtype, domain_dtype, atol, rtol,
                         only_r_linear)
    _full_implementation(op.adjoint.inverse, domain_dtype, target_dtype, atol,
                         rtol, only_r_linear)


def _get_acceptable_location(op, loc, lin):
    if not np.isfinite(lin.val.sum()):
        raise ValueError('Initial value must be finite')
    dir = from_random("normal", loc.domain)
    dirder = lin.jac(dir)
    if dirder.norm() == 0:
        dir = dir * (lin.val.norm()*1e-5)
    else:
        dir = dir * (lin.val.norm()*1e-5/dirder.norm())
    # Find a step length that leads to a "reasonable" location
    for i in range(50):
        try:
            loc2 = loc+dir
            lin2 = op(Linearization.make_var(loc2, lin.want_metric))
            if np.isfinite(lin2.val.sum()) and abs(lin2.val.sum()) < 1e20:
                break
        except FloatingPointError:
            pass
        dir = dir*0.5
    else:
        raise ValueError("could not find a reasonable initial step")
    return loc2, lin2


def check_jacobian_consistency(op, loc, tol=1e-8, ntries=100):
    """
    Checks the Jacobian of an operator against its finite difference
    approximation.

    Computes the Jacobian with finite differences and compares it to the
    implemented Jacobian.

    Parameters
    ----------
    op : Operator
        Operator which shall be checked.
    loc : Field or MultiField
        An Field or MultiField instance which has the same domain
        as op. The location at which the gradient is checked
    tol : float
        Tolerance for the check.
    """
    _domain_check(op)
    _actual_domain_check_nonlinear(op, loc)
    for _ in range(ntries):
        lin = op(Linearization.make_var(loc))
        loc2, lin2 = _get_acceptable_location(op, loc, lin)
        dir = loc2-loc
        locnext = loc2
        dirnorm = dir.norm()
        for i in range(50):
            locmid = loc + 0.5*dir
            linmid = op(Linearization.make_var(locmid))
            dirder = linmid.jac(dir)
            numgrad = (lin2.val-lin.val)
            xtol = tol * dirder.norm() / np.sqrt(dirder.size)
            if (abs(numgrad-dirder) <= xtol).all():
                break
            dir = dir*0.5
            dirnorm *= 0.5
            loc2, lin2 = locmid, linmid
        else:
            raise ValueError("gradient and value seem inconsistent")
        loc = locnext
