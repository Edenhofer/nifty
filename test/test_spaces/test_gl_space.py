# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import itertools

import numpy as np
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_equal,
                           assert_raises)

from nifty6 import GLSpace

pmp = pytest.mark.parametrize

# [nlat, nlon, expected]
CONSTRUCTOR_CONFIGS = [[
    2, None, {
        'nlat': 2,
        'nlon': 3,
        'harmonic': False,
        'shape': (6,),
        'size': 6,
    }
], [0, None, {
    'error': ValueError
}]]


def get_dvol_configs():
    np.random.seed(42)
    wgt = [2.0943951, 2.0943951]
    # for GLSpace(nlat=2, nlon=3)
    dvol_0 = np.array(
        list(
            itertools.chain.from_iterable(
                itertools.repeat(x, 3) for x in wgt)))
    return [
        [1, dvol_0],
    ]


@pmp('attribute', ['nlat', 'nlon'])
def test_property_ret_type(attribute):
    g = GLSpace(2)
    assert_(isinstance(getattr(g, attribute), int))


@pmp('nlat, nlon, expected', CONSTRUCTOR_CONFIGS)
def test_constructor(nlat, nlon, expected):
    g = GLSpace(4)

    if 'error' in expected:
        with assert_raises(expected['error']):
            GLSpace(nlat, nlon)
    else:
        g = GLSpace(nlat, nlon)
        for key, value in expected.items():
            assert_equal(getattr(g, key), value)


@pmp('power, expected', get_dvol_configs())
def test_dvol(power, expected):
    assert_almost_equal(GLSpace(2).dvol, expected)
