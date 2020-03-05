# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises

import nifty6 as ift

pmp = pytest.mark.parametrize
# [lmax, expected]
CONSTRUCTOR_CONFIGS = [[
    5, None, {
        'lmax': 5,
        'mmax': 5,
        'shape': (36,),
        'harmonic': True,
        'size': 36,
    }
], [
    7, 4, {
        'lmax': 7,
        'mmax': 4,
        'shape': (52,),
        'harmonic': True,
        'size': 52,
    }
], [-1, 28, {
    'error': ValueError
}]]


def _k_length_array_helper(index_arr, lmax):
    if index_arr <= lmax:
        index_half = index_arr
    else:
        if (index_arr - lmax) % 2 == 0:
            index_half = (index_arr + lmax)//2
        else:
            index_half = (index_arr + lmax + 1)//2

    m = np.ceil(((2*lmax + 1) - np.sqrt((2*lmax + 1)**2 - 8 *
                                        (index_half - lmax)))/2).astype(int)

    return index_half - m*(2*lmax + 1 - m)//2


def get_k_length_array_configs():
    da_0 = [_k_length_array_helper(idx, 5) for idx in np.arange(36)]
    return [[5, da_0]]


@pmp('attribute', ['lmax', 'mmax', 'size'])
def test_property_ret_type(attribute):
    assert_(isinstance(getattr(ift.LMSpace(7, 5), attribute), int))


@pmp('lmax, mmax, expected', CONSTRUCTOR_CONFIGS)
def test_constructor(lmax, mmax, expected):
    if 'error' in expected:
        with assert_raises(expected['error']):
            ift.LMSpace(lmax, mmax)
    else:
        for key, value in expected.items():
            assert_equal(getattr(ift.LMSpace(lmax, mmax), key), value)


def test_dvol():
    assert_allclose(ift.LMSpace(5).dvol, 1.)


@pmp('lmax, expected', get_k_length_array_configs())
def test_k_length_array(lmax, expected):
    assert_allclose(ift.LMSpace(lmax).get_k_length_array().val, expected)
