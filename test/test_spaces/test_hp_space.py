# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2019 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_equal,
                           assert_raises)

from nifty6 import HPSpace

pmp = pytest.mark.parametrize
# [nside, expected]
CONSTRUCTOR_CONFIGS = [[
    2, {
        'nside': 2,
        'harmonic': False,
        'shape': (48,),
        'size': 48,
    }
], [5, {
    'nside': 5,
    'harmonic': False,
    'shape': (300,),
    'size': 300,
}], [1, {
    'nside': 1,
    'harmonic': False,
    'shape': (12,),
    'size': 12,
}], [0, {
    'error': ValueError
}]]


def test_property_ret_type():
    x = HPSpace(2)
    assert_(isinstance(getattr(x, 'nside'), int))


@pmp('nside, expected', CONSTRUCTOR_CONFIGS)
def test_constructor(nside, expected):
    if 'error' in expected:
        with assert_raises(expected['error']):
            HPSpace(nside)
    else:
        h = HPSpace(nside)
        for key, value in expected.items():
            assert_equal(getattr(h, key), value)


def test_dvol():
    assert_almost_equal(HPSpace(2).dvol, np.pi/12)
