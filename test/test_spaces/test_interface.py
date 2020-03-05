# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from types import LambdaType

import pytest
from numpy.testing import assert_

import nifty6 as ift

pmp = pytest.mark.parametrize


@pmp('attr_expected_type',
     [['harmonic', bool], ['shape', tuple], ['size', int]])
@pmp('space', [
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])
def test_property_ret_type(space, attr_expected_type):
    assert_(
        isinstance(
            getattr(space, attr_expected_type[0]), attr_expected_type[1]))


@pmp('method_expected_type',
     [['get_k_length_array', ift.Field],
      ['get_fft_smoothing_kernel_function', 2.0, LambdaType]])
@pmp('space', [ift.RGSpace(4, harmonic=True), ift.LMSpace(5)])
def test_method_ret_type(space, method_expected_type):
    assert_(
        type(
            getattr(space, method_expected_type[0])
            (*method_expected_type[1:-1])) is method_expected_type[-1])
