# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

# In NIFTy, users can add hand-crafted point-wise nonlinearities that are then
# available for `Field`, `MultiField`, `Linearization` and `Operator`. This
# guide illustrates how this is done.

# Suppose that we would like to use the point-wise function f(x) = x*exp(x) in
# an operator chain. This function is called "myptw" in the following. We
# introduce this function to NIFTy by implementing two functions.

# First, one that takes a `numpy.ndarray` as an input, applies the point-wise
# mapping and returns the result as a `numpy.ndarray` of the same shape.
# Second, a function that takes a `numpy.ndarray` as an input and returns two
# `numpy.ndarray`s: the application of the nonlinearity (same as before) and
# the derivative.


def func(x):
    return x*np.exp(x)


def func_and_derv(x):
    expx = np.exp(x)
    return x*expx, (1+x)*expx


# These two functions are then added to the NIFTy-internal dictionary that
# contains all implemented point-wise nonlinearities.

ift.pointwise.ptw_dict["myptw"] = func, func_and_derv

# This allows us to apply this non-linearity on `Field`s, ...

dom = ift.UnstructuredDomain(10)
fld = ift.from_random(dom)
fld = ift.full(dom, 2.)
a = fld.ptw("myptw")
b = ift.makeField(dom, func(fld.val))
ift.extra.assert_allclose(a, b)

# `MultiField`s, ...
mdom = ift.makeDomain({"bar": ift.UnstructuredDomain(10)})
mfld = ift.from_random(mdom)
a = mfld.ptw("myptw")
b = ift.makeField(mdom, {"bar": func(mfld["bar"].val)})
ift.extra.assert_allclose(a, b)

# Linearizations (including the Jacobian), ...
# (Value)
lin = ift.Linearization.make_var(fld)
a = lin.ptw("myptw").val
b = ift.makeField(dom, func(fld.val))
ift.extra.assert_allclose(a, b)
# (Jacobian)
op_a = lin.ptw("myptw").jac
op_b = ift.makeOp(ift.makeField(dom, func_and_derv(fld.val)[1]))
testing_vector = ift.from_random(dom)
ift.extra.assert_allclose(op_a(testing_vector),
                          op_b(testing_vector))

# and `Operator`s.
op = ift.FieldAdapter(dom, "foo").ptw("myptw")

# We check that the gradient has been implemented correctly by comparing it to
# an approximation to the gradient by finite differences.
def check(func_name, eps=1e-7):
    pos = ift.from_random(ift.UnstructuredDomain(10))
    var0 = ift.Linearization.make_var(pos)
    var1 = ift.Linearization.make_var(pos+eps)
    df0 = (var1.ptw(func_name).val - var0.ptw(func_name).val)/eps
    df1 = var0.ptw(func_name).jac(ift.full(lin.domain, 1.))
    # rtol depends on how nonlinear the function is
    ift.extra.assert_allclose(df0, df1, rtol=100*eps)

check("myptw")
